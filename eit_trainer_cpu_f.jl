# ENV["ROCM_PATH"] = "/usr"
# ENV["LD_LIBRARY_PATH"] = "/usr/lib64:" * get(ENV, "LD_LIBRARY_PATH", "")
ENV["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0"

include("eit_normalizer.jl")

using Lux
using MLUtils, Zygote, Optimisers
using Statistics, Random
using CSV, DataFrames
using Printf
using AMDGPU
using JLD2
AMDGPU.functional()
AMDGPU.functional(:MIOpen)

# Hyperparameters
const SIMULATIONS_DIR  = "simulations"
const CHECKPOINT_DIR   = "checkpoints"
const N_SIMULATIONS    = 10_000

const TRAIN_FRAC       = 0.80f0
const VAL_FRAC         = 0.10f0
# test = remainder (1 - TRAIN_FRAC - VAL_FRAC)

const BATCHSIZE        = 512
const MAX_EPOCH        = 500    # ceiling; early stopping usually halts sooner
const LEARNING_RATE    = 1f-3   # initial LR; decays via cosine schedule
const LR_MIN           = 1f-5   # floor of the cosine schedule
const NUM_ACCUM        = 4
const CHECKPOINT_EVERY = 25     # save every N epochs
const PATIENCE         = 30     # early stop after this many epochs w/o val improvement
const DROPOUT          = 0.10f0 # dropout prob on the wide layers
const RNG              = MersenneTwister(1234)

dev_gpu = cpu_device()
dev_cpu = cpu_device()

function load_dataset(sim_dir::String, n::Int)
    x_cols = Vector{Vector{Float32}}()
    y_cols = Vector{Vector{Float32}}()

    skipped = 0
    for id in 1:n
        v_path = joinpath(sim_dir, "simulation_$(id)_voltage.csv")
        m_path = joinpath(sim_dir, "simulation_$(id)_meta.csv")

        if !isfile(v_path) || !isfile(m_path)
            skipped += 1
            continue
        end

        v_df = CSV.read(v_path, DataFrame)
        m_df = CSV.read(m_path, DataFrame)

        push!(x_cols, Float32.(v_df.re)) # 208
        # push!(x_cols, vcat(Float32.(v_df.re), Float32.(v_df.im))) # 416 
        push!(y_cols, Float32[
            m_df.thickness_skin[1],
            m_df.thickness_fat[1],
            m_df.thickness_muscle[1],
            m_df.thickness_bone[1],
        ])
    end

    skipped > 0 && @warn "Skipped $skipped simulations (missing files)"

    x = reduce(hcat, x_cols) # (208, N)
    y = reduce(hcat, y_cols) # (4,   N)
    return x, y
end

function split_data(x::Matrix{Float32}, y::Matrix{Float32};
                    train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, rng=RNG)
    n        = size(x, 2)
    idx      = randperm(rng, n)
    n_train  = round(Int, train_frac * n)
    n_val    = round(Int, val_frac   * n)

    train_idx = idx[1:n_train]
    val_idx   = idx[n_train+1:n_train+n_val]
    test_idx  = idx[n_train+n_val+1:end]

    return (x[:, train_idx], y[:, train_idx]),
           (x[:, val_idx],   y[:, val_idx]),
           (x[:, test_idx],  y[:, test_idx])
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

# Loss (MSE on normalized outputs)
function loss_fn(model, ps, st, x, y)
    ŷ, st_new = model(x, ps, st)
    loss = mean((ŷ .- y) .^ 2)
    return loss, st_new
end

# Returns MSE loss + MAE per tissue in mm
function validate(model, ps, st, x, y)
    st_test = Lux.testmode(st)
    ŷ, _ = model(x, ps, st_test)
    loss = mean((ŷ .- y) .^ 2)

    ŷ_cpu = Array(ŷ)
    y_cpu = Array(y)
    mae_mm = vec(mean(abs.(denormalize_y(ŷ_cpu) .- denormalize_y(y_cpu)), dims=2)) .* 1f3

    return loss, mae_mm
end

function test_metrics(model, ps, st, x, y)
    st_test = Lux.testmode(st)
    ŷ, _ = model(x, ps, st_test)

    ŷ_mm = denormalize_y(Array(ŷ)) .* 1f3   # > mm
    y_mm = denormalize_y(Array(y)) .* 1f3

    err    = ŷ_mm .- y_mm
    mae    = vec(mean(abs.(err), dims=2))
    rmse   = vec(sqrt.(mean(err .^ 2, dims=2)))

    # per-tissue R²
    ȳ      = mean(y_mm, dims=2)
    ss_res = vec(sum(err .^ 2, dims=2))
    ss_tot = vec(sum((y_mm .- ȳ) .^ 2, dims=2))
    r2     = 1f0 .- ss_res ./ ss_tot

    overall_mse = mean((Array(ŷ) .- Array(y)) .^ 2) # normalized-scale MSE
    return overall_mse, mae, rmse, r2
end

# Checkpoint helpers
function save_checkpoint(ps, st, opt_state, epoch, val_loss, norm, dir)
    mkpath(dir)
    path = joinpath(dir, "checkpoint_epoch_$(lpad(epoch, 4, '0')).jld2")
    @save path ps st opt_state epoch val_loss norm
    @info "Checkpoint saved > $path"
end

function save_best(ps, st, norm, dir)
    mkpath(dir)
    path = joinpath(dir, "best_model.jld2")
    @save path ps st norm
    @info "Best model saved > $path"
end

function main()
    @info "Loading dataset…"
    x, y = load_dataset(SIMULATIONS_DIR, N_SIMULATIONS)
    @info "Loaded $(size(x, 2)) simulations | input=$(size(x,1))  output=$(size(y,1))"

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(x, y)
    @info "Split > train=$(size(x_train,2))  val=$(size(x_val,2))  test=$(size(x_test,2))"

    norm     = fit_normalizer(x_train)
    x_train  = normalize_x(x_train, norm)
    x_val    = normalize_x(x_val,   norm)
    x_test   = normalize_x(x_test,  norm)
    y_train  = normalize_y(y_train)
    y_val    = normalize_y(y_val)
    y_test   = normalize_y(y_test)

    @info "x_train stats" min=minimum(x_train) max=maximum(x_train) mean=mean(x_train)
    @info "y_train stats" min=minimum(y_train) max=maximum(y_train)
    @info "non-finite in x_train" count=count(!isfinite, x_train)
    @info "non-finite in y_train" count=count(!isfinite, y_train)

    # x_train_gpu = x_train |> dev_gpu
    # y_train_gpu = y_train |> dev_gpu
    x_val_gpu   = x_val   |> dev_gpu
    y_val_gpu   = y_val   |> dev_gpu
    x_test_gpu  = x_test  |> dev_gpu
    y_test_gpu  = y_test  |> dev_gpu

    model     = build_model(size(x_train, 1), size(y_train, 1))
    ps, st    = Lux.setup(RNG, model) |> dev_gpu

    train_loader = DataLoader((x_train, y_train), batchsize=BATCHSIZE, shuffle=true)

    rule = OptimiserChain(AccumGrad(NUM_ACCUM), Optimisers.Adam(LEARNING_RATE))
    opt_state = Optimisers.setup(rule, ps)

    # Cosine annealing schedule: LR goes LEARNING_RATE > LR_MIN over MAX_EPOCH.
    cosine_lr(epoch) = LR_MIN + 0.5f0 * (LEARNING_RATE - LR_MIN) *
                       (1f0 + cos(Float32(π) * (epoch - 1) / MAX_EPOCH))

    best_val_loss = Inf32
    epochs_no_improve = 0
    @info "Starting training for up to $MAX_EPOCH epochs (early stop patience=$PATIENCE)…"
    @printf("\n%-6s  %-9s  %-12s  %-12s  %-10s  %-10s  %-10s  %-10s\n",
            "Epoch", "LR", "Train MSE", "Val MSE", "skin mm", "fat mm", "muscle mm", "bone mm")
    println(repeat("-", 92))

    diverged = false

    for epoch in 1:MAX_EPOCH
        # Apply cosine LR schedule for this epoch.
        lr_now = cosine_lr(epoch)
        Optimisers.adjust!(opt_state, lr_now)

        epoch_loss = 0f0

        for (x_b_cpu, y_b_cpu) in train_loader
            x_b = x_b_cpu |> dev_gpu
            y_b = y_b_cpu |> dev_gpu
            (loss_val, st_new), back = Zygote.pullback(
                p -> loss_fn(model, p, st, x_b, y_b), ps
            )

            if !isfinite(loss_val)
                @error "Non-finite batch loss at epoch $epoch — aborting training. Last good checkpoint is preserved."
                diverged = true
                break
            end

            grads = back((one(loss_val), nothing))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            st = st_new
            epoch_loss += loss_val
        end

        diverged && break

        epoch_loss /= length(train_loader)
        # Backup guard at the epoch level (in case an Inf averaged to something odd).
        if !isfinite(epoch_loss)
            @error "Non-finite epoch loss at epoch $epoch — aborting training."
            break
        end

        val_loss, mae_mm = validate(model, ps, st, x_val_gpu, y_val_gpu)

        @printf("%-6d  %-9.2e  %-12.6f  %-12.6f  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n",
                epoch, lr_now, epoch_loss, val_loss,
                mae_mm[1], mae_mm[2], mae_mm[3], mae_mm[4])

        # Save best
        if isfinite(val_loss) && val_loss < best_val_loss
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_best(ps |> dev_cpu, st |> dev_cpu, norm, CHECKPOINT_DIR)
        else
            epochs_no_improve += 1
        end

        # Periodic checkpoint
        if epoch % CHECKPOINT_EVERY == 0
            save_checkpoint(ps |> dev_cpu, st |> dev_cpu, opt_state,
                            epoch, val_loss, norm, CHECKPOINT_DIR)
        end

        # Early stopping
        if epochs_no_improve >= PATIENCE
            @info "Early stopping at epoch $epoch — no val improvement for $PATIENCE epochs (best val MSE=$(round(best_val_loss, sigdigits=5)))."
            break
        end
    end

    println(repeat("-", 76))
    best_path = joinpath(CHECKPOINT_DIR, "best_model.jld2")
    if isfile(best_path)
        @info "Reloading best checkpoint for final test > $best_path"
        ps_best = ps; st_best = st
        JLD2.@load best_path ps st
        ps = ps |> dev_gpu
        st = st |> dev_gpu
    end
    test_mse, test_mae, test_rmse, test_r2 = test_metrics(model, ps, st, x_test_gpu, y_test_gpu)
    @printf("\nTest MSE (normalized): %.6f\n\n", test_mse)
    @printf("%-8s  %-10s  %-10s  %-8s\n", "tissue", "MAE (mm)", "RMSE (mm)", "R²")
    println(repeat("-", 42))
    for (i, name) in enumerate(TISSUE_NAMES)
        @printf("%-8s  %-10.4f  %-10.4f  %-8.4f\n",
                name, test_mae[i], test_rmse[i], test_r2[i])
    end
    @printf("\nMean R² across tissues: %.4f\n", mean(test_r2))
end

main()
