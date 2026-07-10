# Command-line inference for the trained EIT arm model.

# Usage:
#	julia --project=. eit_predict.jl sample_data/sim1/simulation_1_voltage.csv --checkpoint checkpoints/complex/best_model.jld2 --dims 416 --samples 200 --out result.png
#	julia --project=. eit_predict.jl sample_data/sim1/simulation_1_voltage.csv --checkpoint checkpoints/real_only/best_model.jld2 --dims 208 --samples 200 --out result.png

# The CSV must contain the re and im voltage columns (416 features for complex model or 208 re-only for real model).

# Output:
#	prints skin / fat / muscle / bone thickness in mm
#	prints an MC-dropout uncertainty
#	saves an arm cross-section diagram (concentric tissue layers) as a PNG

using Lux
using JLD2
using CSV, DataFrames
using Statistics, Random
using Printf
using CairoMakie

include("eit_normalizer.jl") # for EITNormalizer, normalize_x, denormalize_y, Y_MIN/MAX, TISSUE_NAMES

const DROPOUT    = 0.10f0 # must match training
const OUT_DIMS   = 4

function parse_args(args)
    isempty(args) && error("Usage: julia eit_predict.jl <voltage.csv> [--checkpoint path] [--samples N] [--out file.png] [--dims 208|416]")
    csv_path   = args[1]
    samples    = 200
    outfile    = "arm_prediction.png"
    in_dims    = 416
    checkpoint = "checkpoints_complex_399/best_model.jld2" # default to best model
    i = 2
    while i <= length(args)
        if args[i] == "--samples"
            samples = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--out"
            outfile = args[i+1]; i += 2
        elseif args[i] == "--dims"
            in_dims = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--checkpoint"
            checkpoint = args[i+1]; i += 2
        else
            error("Unknown argument: $(args[i])")
        end
    end
    return csv_path, samples, outfile, in_dims, checkpoint
end

function build_model(in_dims::Int, out_dims::Int)
    return Chain(
        Dense(in_dims => 512),
        BatchNorm(512, relu),
        Dropout(DROPOUT),
        Dense(512 => 256),
        BatchNorm(256, relu),
        Dropout(DROPOUT),
        Dense(256 => 128),
        BatchNorm(128, relu),
        Dense(128 => 64,  relu),
        Dense(64  => out_dims),
    )
end

function read_voltage(path::String, in_dims::Int)::Matrix{Float32}
    isfile(path) || error("File not found: $path")
    df = CSV.read(path, DataFrame)

    vals = if in_dims == 208 && "re" in names(df)
        # real-only model — use only re column
        if "im" in names(df)
            @warn "CSV has `im` column but --dims 208 was specified. Only `re` will be used."
        end
        Float32.(df.re)
    elseif "re" in names(df) && "im" in names(df)
        vcat(Float32.(df.re), Float32.(df.im)) # 416 features (re + im)
    elseif ncol(df) == 1
        Float32.(df[!, 1])
    elseif nrow(df) == 1
        Float32.(collect(df[1, :]))
    else
        error("Could not find voltage data.")
    end

    length(vals) == in_dims ||
        error("Expected $in_dims voltages, got $(length(vals)).")
    return reshape(vals, in_dims, 1)
end

# Predict: one clean (testmode) pass for the point estimate, plus N stochastic
# (dropout-on) passes for MC-dropout uncertainty. Returns mm values.
function predict(model, ps, st, norm, x_raw::Matrix{Float32}, n_samples::Int)
    x_n = normalize_x(x_raw, norm)

    st_test = Lux.testmode(st)
    ŷ_n, _  = model(x_n, ps, st_test)
    point_mm = vec(denormalize_y(Array(ŷ_n))) .* 1f3

    # st_train = make_dropout_active(Lux.testmode(st))
    # @show st_train
    tile = 64
    x_tiled = repeat(x_n, 1, tile) # (in_dims, tile), all columns identical
    samples = Matrix{Float32}(undef, OUT_DIMS, n_samples)
    for s in 1:n_samples
        st_s = activate_dropout(st_test)
        ŷs, _ = model(x_tiled, ps, st_s)
        col   = vec(mean(denormalize_y(Array(ŷs)), dims=2)) .* 1f3
        samples[:, s] = col
    end
    std_mm = vec(std(samples, dims=2))

    return point_mm, std_mm
end

# function make_dropout_active(st)
#     return Lux.update_state(st, :active, Val(true))
# end

function activate_dropout(st)
    layers = keys(st)
    new_st = NamedTuple{layers}(map(layers) do k
        s = st[k]
        if haskey(s, :training) && haskey(s, :rng)
            (rng = Random.MersenneTwister(rand(UInt32)), training = Val(true))
        else
            s
        end
    end)
    return new_st
end

# Arm cross-section diagram: concentric layers bone, muscle, fat, skin.
function draw_arm(point_mm, std_mm, outfile)
    skin, fat, muscle, bone = point_mm
    
    r_bone   = bone
    r_muscle = r_bone   + muscle
    r_fat    = r_muscle + fat
    r_skin   = r_fat    + skin

    colors = (bone   = (:antiquewhite, 1.0),
              muscle = (:indianred,    1.0),
              fat    = (:khaki,        1.0),
              skin   = (:peachpuff,    1.0))

    fig = Figure(size = (760, 460))
    ax  = Axis(fig[1, 1], aspect = DataAspect(),
               title = "Predicted arm cross-section (radial, mm)")
    hidedecorations!(ax); hidespines!(ax)

    poly!(ax, Circle(Point2f(0, 0), r_skin),   color = colors.skin[1])
    poly!(ax, Circle(Point2f(0, 0), r_fat),    color = colors.fat[1])
    poly!(ax, Circle(Point2f(0, 0), r_muscle), color = colors.muscle[1])
    poly!(ax, Circle(Point2f(0, 0), r_bone),   color = colors.bone[1])

    band_lo = r_skin - std_mm[1]
    band_hi = r_skin + std_mm[1]
    lines!(ax, Circle(Point2f(0, 0), band_hi), color = (:black, 0.25), linestyle = :dash)
    lines!(ax, Circle(Point2f(0, 0), band_lo), color = (:black, 0.25), linestyle = :dash)

    lim = r_skin * 1.15
    limits!(ax, -lim, lim, -lim, lim)

    labels = ["skin",  "fat",   "muscle", "bone"]
    cols   = [colors.skin[1], colors.fat[1], colors.muscle[1], colors.bone[1]]
    elems  = [PolyElement(color = c) for c in cols]
    leg_txt = [@sprintf("%-7s %.2f ± %.2f mm", labels[i], point_mm[i], std_mm[i]) for i in 1:4]
    Legend(fig[1, 2], elems, leg_txt, "Tissue (point ± MC-dropout σ)",
           framevisible = true, labelsize = 14)

    save(outfile, fig)
    return outfile
end

function main()
    csv_path, n_samples, outfile, in_dims, checkpoint = parse_args(ARGS)

    isfile(checkpoint) || error("Checkpoint not found: $checkpoint (train a model first).")
    @info "Loading model" checkpoint=checkpoint
    ps = nothing; st = nothing; norm = nothing
    JLD2.@load checkpoint ps st norm

    model = build_model(in_dims, OUT_DIMS)

    @info "Reading voltages" file=csv_path
    x_raw = read_voltage(csv_path, in_dims)

    @info "Predicting" mc_samples=n_samples
    point_mm, std_mm = predict(model, ps, st, norm, x_raw, n_samples)

    println("\nPredicted tissue thickness (point estimate ± MC-dropout σ):")
    println(repeat("-", 48))
    for i in 1:OUT_DIMS
        @printf("  %-8s %6.3f ± %5.3f mm\n", TISSUE_NAMES[i], point_mm[i], std_mm[i])
    end
    @printf("\n  total radius  %6.3f mm\n", sum(point_mm))

    saved = draw_arm(point_mm, std_mm, outfile)
    @info "Diagram saved" file=saved
end

main()
