# Normalization for the EIT arm dataset.

# Inputs  (voltage): z-score per feature
# Outputs (thickness): min-max to [0,1] - bounds from the known sampling ranges

using Statistics, Printf

# Order: [skin, fat, muscle, bone]
const Y_MIN = Float32[1e-3, 2e-3,  15e-3,  8e-3]
const Y_MAX = Float32[3e-3, 35e-3, 55e-3, 22e-3]

# Normalizer struct
struct EITNormalizer
    x_mean :: Vector{Float32} # (n_features,)
    x_std  :: Vector{Float32} # (n_features,)
end

# Fit — call on training set only, then apply to val/test
# x shape: (n_features, n_samples)
function fit_normalizer(x::Matrix{Float32}) :: EITNormalizer
    μ = vec(mean(x, dims=2))
    σ = vec(std(x,  dims=2))
    σ = max.(σ, 1f-8) # guard against zero-variance features
    return EITNormalizer(μ, σ)
end

# Apply / invert input normalization
function normalize_x(x::Matrix{Float32}, norm::EITNormalizer) :: Matrix{Float32}
    return (x .- norm.x_mean) ./ norm.x_std
end

function denormalize_x(x_n::Matrix{Float32}, norm::EITNormalizer) :: Matrix{Float32}
    return x_n .* norm.x_std .+ norm.x_mean
end

# Apply / invert output normalization
# y shape: (4, n_samples)  — rows are [skin, fat, muscle, bone] in metres
function normalize_y(y::Matrix{Float32}) :: Matrix{Float32}
    return (y .- Y_MIN) ./ (Y_MAX .- Y_MIN)
end

function denormalize_y(y_n::Matrix{Float32}) :: Matrix{Float32}
    return y_n .* (Y_MAX .- Y_MIN) .+ Y_MIN
end

# Convenience: MAE per tissue in mm (for readable training logs)
const TISSUE_NAMES = ["skin", "fat", "muscle", "bone"]

function mae_per_tissue(y_pred_n::Matrix{Float32}, y_true_n::Matrix{Float32})
    y_pred = denormalize_y(y_pred_n)
    y_true = denormalize_y(y_true_n)
    mae_mm = vec(mean(abs.(y_pred .- y_true), dims=2)) .* 1f3   # metres > mm
    for (name, val) in zip(TISSUE_NAMES, mae_mm)
        @printf("  MAE %-8s %.4f mm\n", name, val)
    end
    return mae_mm
end
