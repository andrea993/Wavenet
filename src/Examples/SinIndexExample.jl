# =====================================================================================
# SineIndexExample.jl — Didactic end-to-end example using the Wavenet library
#
# This script reproduces your original steps:
#   - Build a sine dataset y[k] = sin(2π f t[k]) with L samples on [0,T)
#   - Randomly split indices into learning (20%) and validation (80%)
#   - Scale inputs to the wavelet support
#   - Build lattice candidates via `find_covering_wavelets`
#   - Stepwise selection to initialize output weights
#   - Fine-tune candidates + output weights with SGD
#   - Evaluate initial and fine-tuned models across the full grid
#   - Plot dataset vs. initial vs. fine-tuned predictions
# =====================================================================================

module SineIndexExample

include("../Wavenet.jl")

using .Wavenet
using Plots
using Infiltrator
using Random

# ------------------------ Helpers ------------------------

# Domain helper (per-row min/max with padding), for X as d×N
# Note: in this example we use a 1D index domain, but we keep the generic helper.
function _domain_from_data(X::AbstractMatrix{<:Real}; pad=0.2)
    mins = vec(minimum(X, dims=2))
    maxs = vec(maximum(X, dims=2))
    span = maxs .- mins
    lo   = mins .- pad .* span
    hi   = maxs .+ pad .* span
    [range(lo[i], hi[i], length=2) for i in 1:size(X,1)]
end

# Stack translations from candidates into a d×M matrix (handles M==1)
stack_t(cands::AbstractVector{<:Wavenet.WaveletCandidate}) = begin
    T = hcat((c.t for c in cands)...)
    ndims(T) == 1 ? reshape(T, :, 1) : T
end

# ------------------------ Data generation (sine on discrete indices) ------------------------

"""
    generate_sine_index_data(; T=1.0, L=1024, freq=2.0)

Returns:
- t :: Vector{Float64}  — time grid on [0, T)
- y :: Vector{Float64}  — sine samples y = sin(2π*freq*t)
- idx :: Vector{Int}    — 1:L indices (useful as "inputs" for 1D task)

This mirrors your original setup (T=1, L=1024, frequency 2 Hz).
"""
function generate_sine_index_data(; T::Real=1.0, L::Int=1024, freq::Real=2.0)
    dt = T / L
    t  = collect((0:L-1) .* dt)
    y  = sin.(2π * freq .* t)
    idx = collect(1:L)
    return t, y, idx
end

# ------------------------ Training / Prediction ------------------------

"""
    train_sine_wavenet(y, idx; learn_ratio=0.2, s=5, epochs=10_000, batch=64, lr=0.1, l2=1e-4,
                       dilatation=-2:8, support_scale=0.5, seed=123)

Trains a Wavenet to fit y[k] using input = scaled index k. Steps:
1) Shuffle indices and split into learning/validation (by `learn_ratio`).
2) Scale inputs (indices) to the wavelet support.
3) Build lattice candidates via `find_covering_wavelets`.
4) Stepwise selection to initialize output weights.
5) Fine-tune (SGD) candidates and output weights.

Returns:
- Wavelet :: ActivationFunction
- w       :: Vector{Float64}      (final output weights)
- wparams :: Vector{Float64}      (final dilations)
- tparams :: Matrix{Float64}      (final translations, d×M)
- Ws_sel  :: Vector{WaveletCandidate} (selected wavelets from lattice)
- Ws_all  :: Vector{WaveletCandidate} (all lattice wavelets)
- Xs_learn:: Matrix{Float64}      (scaled inputs for learning set, 1×Nlearn)
- domain  :: Vector{UnitRange}    (original domain used for scaling)
- support :: AbstractRange         (activation support)
- y_learn :: Vector{Float64}      (targets used for learning)
- idx_learn :: Vector{Int}        (indices used for learning)
- idx_full  :: Vector{Float64}    (scaled full index grid 1×L as vector)
- y_init    :: Vector{Float64}    (initial model over full grid)
"""
function train_sine_wavenet(
    y::Vector{<:Real}, idx::Vector{<:Integer};
    learn_ratio::Real=0.2, s::Int=5,
    epochs::Int=10_000, batch::Int=64, lr::Real=0.1, l2::Real=1e-4,
    dilatation=-2:8, support_scale::Real=0.5,
    seed::Integer=123
)
    @assert length(y) == length(idx) "y and idx length mismatch."
    L = length(idx)
    Random.seed!(seed)

    # Activation and support
    Wavelet = Wavenet.RadialActivation.MexicanHat
    support = Wavelet.support

    # Shuffle indices and split learning/validation
    idxs = shuffle(idx)
    learn_L = Int(floor(L * learn_ratio))
    idx_learn = idxs[1:learn_L]
    idx_valid = idxs[learn_L+1:end]

    y_learn = y[idx_learn]
    y_valid = y[idx_valid]

    # Domain is just 1:L (1D index input), but keep as Vector of ranges to use scaling helper
    domain = [1:L]

    # Scale inputs: (1×N) matrices expected by the library
    Xs_learn = Wavenet.data_scale_to_support(reshape(Float64.(idx_learn), 1, :), domain, support)
    idx_scaled_full = Wavenet.data_scale_to_support(reshape(Float64.(idx), 1, :), domain, support)

    # Build lattice candidates on the LEARNING inputs
    # Note: supportScale < 1.0 shrinks the box in lattice construction → fewer candidates
    Ws_all = Wavenet.find_covering_wavelets(Xs_learn, support, dilatation, support_scale)
    @assert !isempty(Ws_all) "No candidate wavelets from covering. Try larger support_scale or adjust dilatation."

    # Stepwise selection over lattice candidates
    (w_init, sel_idx, A, Q) = Wavenet.stepwise_wavelet_selection_wt(
        Wavelet, Ws_all, Xs_learn, y_learn, s)

    Ws_sel = Ws_all[sel_idx]

    # Initial curve over FULL grid (for comparison)
    w0 = [c.w for c in Ws_sel]
    t0 = stack_t(Ws_sel)
    X_full = reshape(idx_scaled_full, 1, :)
    y_init = Wavenet.wavenet_evaluate(Wavelet, w_init, w0, t0, X_full)

    # Fine-tune selected candidates + output weights
    (w, cands, tr_loss, val_loss) = Wavenet.finetune_from_candidates!(
        Wavelet, Ws_sel, w_init, Xs_learn, y_learn;
        epochs=epochs, batch=batch, lr=lr, l2=l2
    )

    # Extract learned params for evaluation
    wparams = [c.w for c in cands]
    tparams = stack_t(cands)

    return Wavelet, w, wparams, tparams, Ws_sel, Ws_all, Xs_learn, domain, support,
           y_learn, idx_learn, vec(idx_scaled_full), y_init
end

"""
    predict_full_grid(Wavelet, w, wparams, tparams, idx_scaled_full)

Evaluates the fine-tuned model on the full (scaled) index grid.
Returns ŷ_full :: Vector{Float64}.
"""
function predict_full_grid(Wavelet, w::Vector{<:Real},
                           wparams::Vector{<:Real}, tparams::Matrix{<:Real},
                           idx_scaled_full::Vector{<:Real})
    X_full = reshape(idx_scaled_full, 1, :)
    return Wavenet.wavenet_evaluate(Wavelet, w, wparams, tparams, X_full)
end

# ------------------------ End-to-end example ------------------------

"""
    run_sine_index_example(; T=1.0, L=1024, freq=2.0,
                            learn_ratio=0.2, s=5,
                            epochs=10_000, batch=64, lr=0.1, l2=1e-4,
                            dilatation=-2:8, support_scale=0.5, seed=123)

Runs the full pipeline and produces a plot with:
- Dataset (ground truth)
- Initial Wavenet (from stepwise selection)
- Fine-tuned Wavenet
"""
function run_sine_index_example(;
    T::Real=1.0, L::Int=1024, freq::Real=2.0,
    learn_ratio::Real=0.2, s::Int=5,
    epochs::Int=10_000, batch::Int=64, lr::Real=0.1, l2::Real=1e-4,
    dilatation=-2:8, support_scale::Real=0.5, seed::Integer=123
)
    # 1) Data
    t, y, idx = generate_sine_index_data(; T=T, L=L, freq=freq)

    # 2) Train
    Wavelet, w, wparams, tparams, Ws_sel, Ws_all, Xs_learn, domain, support,
    y_learn, idx_learn, idx_scaled_full, y_init =
        train_sine_wavenet(y, idx;
            learn_ratio=learn_ratio, s=s, epochs=epochs, batch=batch, lr=lr, l2=l2,
            dilatation=dilatation, support_scale=support_scale, seed=seed)

    # 3) Predict (full grid)
    ŷ_full = predict_full_grid(Wavelet, w, wparams, tparams, idx_scaled_full)

    # 4) Plot (replicates your original final plot)
    p = plot(idx, y_init, label="wavenet (init)", legend=:bottomleft)
    plot!(p, idx, ŷ_full, label="wavenet (finetuned)")
    plot!(p, idx, y, label="dataset")
    title!(p, "Sine over indices — s=$s, epochs=$epochs")
    xlabel!("Index k"); ylabel!("y(k)")
    display(p)

    return p
end

# Auto-run for convenience (comment out if using as a library)
run_sine_index_example()

end # module SineIndexExample
