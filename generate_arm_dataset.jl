using JuliaFEM
using GLMakie
using CSV
using DataFrames
using Random
using ZipFile

GLMakie.closeall()

# Configuration
const N_SIMULATIONS    = 10_000
const BATCH_SIZE       = 500 # simulations per zip archive
const OUTPUT_DIR       = "simulations/arm"
const SEED             = 42

const NUMBER_OF_ELECTRODES = 16
const INJ_CURRENT          = 100e-6 # 100 μA

# Fixed tissue conductivities (S/m) and relative permittivities
const σ_muscle  = 0.516
const σ_bone    = 0.02
const σ_fat     = 0.0435
const σ_skin    = 0.01375

const ε_muscle  = 1867.994
const ε_bone    = 150.0
const ε_fat     = 47.9
const ε_skin    = 1080.083

# outer_radius = t_bone + t_muscle + t_fat + t_skin
# Target outer_radius ≈ 75–90 mm (typical adult forearm/upper arm)

# Layer thickness ranges [min, max] in metres
const SKIN_RANGE   = (1e-3,  3e-3) # 1–3 mm
const FAT_RANGE    = (2e-3, 35e-3) # 2–35 mm
const BONE_RANGE   = (8e-3, 22e-3) # 8–22 mm

# Muscle fills the remainder, we enforce a hard minimum below.
const MUSCLE_MIN   = 15e-3 # at least 15 mm of muscle
const MUSCLE_MAX   = 55e-3

# Latin Hypercube Sampling
function latin_hypercube(n::Int, dims::Int; rng=Random.default_rng())
    samples = zeros(n, dims)
    for d in 1:dims
        perm = randperm(rng, n)
        for i in 1:n
            samples[perm[i], d] = (i - 1 + rand(rng)) / n
        end
    end
    return samples # values in [0, 1]
end

function scale(u, lo, hi)
    return lo + u * (hi - lo)
end

function generate_parameter_sets(n::Int; seed::Int=SEED)
    rng = MersenneTwister(seed)
    lhs = latin_hypercube(n * 2, 3; rng=rng) # oversample 2× to allow rejection

    params = Vector{NamedTuple}()
    sizehint!(params, n)

    for row in eachrow(lhs)
        t_skin = scale(row[1], SKIN_RANGE...)
        t_fat  = scale(row[2], FAT_RANGE...)
        t_bone = scale(row[3], BONE_RANGE...)

        # Muscle fills what's left up to a reasonable total arm radius.
        total_non_muscle = t_skin + t_fat + t_bone
        outer_radius_min = 60e-3
        outer_radius_max = 95e-3
        muscle_lo = max(MUSCLE_MIN, outer_radius_min - total_non_muscle)
        muscle_hi = min(MUSCLE_MAX, outer_radius_max - total_non_muscle)

        if muscle_lo > muscle_hi
            continue
        end

        t_muscle = muscle_lo + rand(rng) * (muscle_hi - muscle_lo)

        push!(params, (
            thickness_skin   = t_skin,
            thickness_fat    = t_fat,
            thickness_muscle = t_muscle,
            thickness_bone   = t_bone,
        ))

        length(params) == n && break
    end

    if length(params) < n
        error("Could not generate $n valid samples after oversampling. " *
              "Adjust the thickness ranges or increase the oversample factor.")
    end

    return params
end

function run_simulation(id::Int, p::NamedTuple, output_dir::String)
    mesh_path = joinpath(output_dir, "smd_$(id).msh")

    generateSkinmodel(
        NUMBER_OF_ELECTRODES,
        p.thickness_skin,
        p.thickness_fat,
        p.thickness_muscle,
        p.thickness_bone;
        name     = mesh_path,
        show_gui = false,
        lc       = 0.01,
    )

    geometry, boundary = getModel(mesh_path)

    fig = Figure(; size = (960, 960))
    ax  = Axis3(fig[1, 1])
    plot_mesh(ax, geometry)
    plot_electrodes(ax, boundary)
    axislegend(ax)
    xlims!(ax, -0.1, 0.1)
    ylims!(ax, -0.1, 0.1)
    save(joinpath(output_dir, "smd_$(id).png"), fig)
    GLMakie.closeall()

    mat = assignMaterial(geometry, [σ_muscle, σ_bone, σ_fat, σ_skin],
                                            [ε_muscle, ε_bone, ε_fat, ε_skin],
                                            [1e6, 1e6, 1e6, 1e6])
    K = buildMatrix(geometry, boundary, mat)
    many_probs = applyBC(K, boundary, INJ_CURRENT)
    many_solution = solve(many_probs)
    many_V = getElectrodeVoltage(many_solution, boundary)

    df = create_simulation_row(
        create_measurement_vector(many_V),
        NUMBER_OF_ELECTRODES,
        p.thickness_skin,
        p.thickness_fat,
        p.thickness_muscle,
        p.thickness_bone,
        mat,
        INJ_CURRENT,
        σ_muscle, σ_bone, σ_fat, σ_skin,
        ε_muscle, ε_bone, ε_fat, ε_skin,
        Val(:arm),
    )

    save_simulation_csv(df, id, Val(:arm); output_dir = output_dir)

    return df
end

function zip_batch(ids::UnitRange{Int}, output_dir::String, archive_dir::String)
    batch_idx  = div(first(ids) - 1, BATCH_SIZE) + 1
    zip_path   = joinpath(archive_dir, "batch_$(lpad(batch_idx, 3, '0')).zip")

    w = ZipFile.Writer(zip_path)
    for id in ids
        for suffix in [
            "smd_$(id).msh",
            "smd_$(id).png",
            "simulation_$(id)_voltage.csv",
            "simulation_$(id)_meta.csv",
            "simulation_$(id)_material.csv",
        ]
            fpath = joinpath(output_dir, suffix)
            if isfile(fpath)
                f = ZipFile.addfile(w, suffix)
                write(f, read(fpath))
                # rm(fpath)
            else
                @warn "Expected file not found, skipping: $fpath"
            end
        end
    end
    close(w)
    @info "Created archive: $zip_path"
end

function main()
    archive_dir = joinpath(OUTPUT_DIR, "archives")
    mkpath(OUTPUT_DIR)
    mkpath(archive_dir)

    @info "Generating $(N_SIMULATIONS) parameter sets with Latin Hypercube Sampling…"
    params = generate_parameter_sets(N_SIMULATIONS; seed = SEED)
    @info "Parameter generation complete. Starting FEM simulations…"

    # Master index accumulator
    index_rows = Vector{NamedTuple}()

    for (i, p) in enumerate(params)
        @info "Simulation $i / $(N_SIMULATIONS)"
        try
            df = run_simulation(i, p, OUTPUT_DIR)
            push!(index_rows, (
                id               = i,
                thickness_skin   = p.thickness_skin,
                thickness_fat    = p.thickness_fat,
                thickness_muscle = p.thickness_muscle,
                thickness_bone   = p.thickness_bone,
                outer_radius_mm  = (p.thickness_skin + p.thickness_fat +
                                    p.thickness_muscle + p.thickness_bone) * 1e3,
            ))
        catch e
            @error "Simulation $i failed: $e"
        end

        # After each complete batch, zip it up
        if i % BATCH_SIZE == 0
            batch_start = i - BATCH_SIZE + 1
            batch_end   = i
            @info "Zipping batch $(div(i, BATCH_SIZE)) (sims $batch_start-$batch_end)…"
            zip_batch(batch_start:batch_end, OUTPUT_DIR, archive_dir)
        end
    end

    # Handle any leftover sims in a partial final batch
    remainder = N_SIMULATIONS % BATCH_SIZE
    if remainder != 0
        batch_start = N_SIMULATIONS - remainder + 1
        @info "Zipping final partial batch (sims $batch_start-$(N_SIMULATIONS))…"
        zip_batch(batch_start:N_SIMULATIONS, OUTPUT_DIR, archive_dir)
    end

    # Write master index CSV
    index_df   = DataFrame(index_rows)
    index_path = joinpath(OUTPUT_DIR, "all_simulations_index.csv")
    CSV.write(index_path, index_df)
    @info "Master index written to: $index_path"
    @info "Done. $(nrow(index_df)) simulations completed successfully."
end

main()
