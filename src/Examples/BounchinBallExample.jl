module BouncingBallExample

include("../Wavenet.jl")

using .Wavenet
using Plots
using Random
using Infiltrator
# -----------------------------
# Physics of the bouncing ball
# -----------------------------
function bounce_step(y::Real, v::Real, dt::Real; g::Real=9.81, e::Real=0.85)
    y2 = y + v*dt - 0.5*g*dt^2
    if y2 >= 0
        return y2, v - g*dt
    else
        disc = v^2 + 2g*y
        τ = (v + sqrt(disc)) / g
        v_hit = v - g*τ
        v_after = -e * v_hit
        dt_rem = dt - τ
        y3 = v_after*dt_rem - 0.5*g*dt_rem^2
        v3 = v_after - g*dt_rem
        return max(0.0, y3), v3
    end
end

function generate_bouncing_ball_data(n_samples::Int=1000, t_end::Real=25.0;
    g=9.81, e=0.85, y0=2.0, v0=0.0)

    dt = t_end / n_samples
    Y = zeros(Float64, n_samples)
    V = zeros(Float64, n_samples)
    y, v = y0, v0
    @inbounds for k in 1:n_samples
        Y[k] = y; V[k] = v
        y, v = bounce_step(y, v, dt; g=g, e=e)
    end
    return Y, V
end

# -----------------------------
# Domain helper
# -----------------------------
function _float_domain_from_data(X::Matrix{Float64}; pad=0.2)
    mins = vec(minimum(X, dims=2))
    maxs = vec(maximum(X, dims=2))
    pd   = pad .* (maxs .- mins)
    [range(mins[i]-pd[i], maxs[i]+pd[i], length=2) for i in 1:size(X,1)]
end

# -----------------------------
# Training
# -----------------------------
function train_bouncing_wavenet_all(X::Matrix{Float64}, y::Vector{Float64};
    s::Int=10, epochs::Int=0, batch::Int=64, lr::Real=0.05, l2::Real=1e-4,
    dilatation=-2:6, pad::Real=0.5, max_train_samples::Int=200)

    N = size(X, 2)

    # --- Choose training subset (random if too many samples) ---
    if N > max_train_samples
        idx = randperm(N)[1:max_train_samples]
        Xsub, ysub = X[:, idx], y[idx]
    else
        Xsub, ysub = X, y
    end

    Wavelet = Wavenet.RadialActivation.MexicanHat
    support = Wavelet.support
    domain  = _float_domain_from_data(X; pad=pad)
    data_scaled = Wavenet.data_scale_to_support(Xsub, domain, support)
    data_scaled .= clamp.(data_scaled, first(support), last(support))  # training only

    W = Wavenet.find_covering_wavelets(data_scaled, support, dilatation, 0.3)
    s_eff = min(s, length(W), size(data_scaled, 2))

    #(w_init, l, A, Q) = Wavenet.stepwise_wavelet_selection_wt(Wavelet, W, data_scaled, ysub, s_eff)
    (w_init, l) = Wavenet.backward_elimination_wt(Wavelet, W, data_scaled, ysub, s_eff)
    
    Ws = W[l]

    w, cands, tr_loss, val_loss = Wavenet.finetune_from_candidates!(
        Wavelet,
        Ws,             # candidates in
        w_init,         # output weights init
        data_scaled,
        ysub;
        epochs=epochs,
        batch=batch,
        lr=lr,
        l2=l2
    )

    wparams = [c.w for c in cands]
    tparams = hcat((c.t for c in cands)...)        # force matrix shape
    if ndims(tparams) == 1; tparams = reshape(tparams, :, 1); end


    
    return Wavelet, w, wparams, tparams, Ws, data_scaled, domain, support
end

# -----------------------------
# Prediction
# -----------------------------
function predict_bouncing_wavenet(Wavelet, w, wparams, tparams,
    X::Matrix{Float64}, domain, support)

    Xs = Wavenet.data_scale_to_support(X, domain, support)

    #@infiltrate
    Wavenet.wavenet_evaluate(Wavelet, w, wparams, tparams, Xs)
end

# -----------------------------
# Rollout simulation
# -----------------------------
function simulate_bouncing_wavenet(model_params, initial_state::Vector{<:Real},
    n_steps::Int, domain, support)

    Wavelet, w, wparams, tparams = model_params
    lag = length(initial_state)
    traj = zeros(Float64, n_steps + lag)
    traj[1:lag] = initial_state
    state = copy(initial_state)

    @inbounds for i in 1:n_steps
        xmat = reshape(state, lag, 1)
        nxt = predict_bouncing_wavenet(Wavelet, w, wparams, tparams, xmat, domain, support)[1]
        traj[lag + i] = nxt
        state[1:end-1] = state[2:end]
        state[end] = nxt
    end
    return traj[lag+1:end]
end

# -----------------------------
# Full run
# -----------------------------
function run_bouncing_ball_example_all(; n_samples=1000, t_end=5.0, g=9.81, e=0.85,
    lag::Int=2, s::Int=10, epochs::Int=1000, batch::Int=64, lr::Real=0.05, l2::Real=1e-4,
    seed=123, dilatation=-1:10, max_train_samples=500)

    Random.seed!(seed)
    Y, _ = generate_bouncing_ball_data(n_samples, t_end; g=g, e=e, y0=2.0, v0=0.0)

    Ncols = n_samples - lag
    X_wavenet = Matrix{Float64}(undef, lag, Ncols)
    y_wavenet = Vector{Float64}(undef, Ncols)
    @inbounds for i in 1:Ncols
        X_wavenet[:, i] = Y[i : i+lag-1]
        y_wavenet[i]    = Y[i+lag]
    end

    Wavelet, α, wparams, tparams, Ws, data_scaled, domain, support =
        train_bouncing_wavenet_all(X_wavenet, y_wavenet;
            s=s, epochs=epochs, batch=batch, lr=lr, l2=l2,
            dilatation=dilatation, max_train_samples=max_train_samples)

    # 1-step ahead
    ŷ_1step = predict_bouncing_wavenet(Wavelet, α, wparams, tparams, X_wavenet, domain, support)

    # rollout
    init_state = X_wavenet[:, 1]
    y_rollout  = simulate_bouncing_wavenet((Wavelet, α, wparams, tparams),
                init_state, Ncols, domain, support)

    idx = (lag+1):n_samples
    p = plot(idx, Y[lag+1:end], label="True trajectory", lw=2)
    plot!(p, idx, ŷ_1step, label="1-step ahead", lw=2, ls=:dash)
    plot!(p, idx, y_rollout, label="Rollout", lw=2, ls=:dot)
    title!(p, "Bouncing Ball (lag=$lag, s=$s, epochs=$epochs)")
    xlabel!("Time step"); ylabel!("Height")
    plot!(p, legend=:topright)
    display(p)
    return p
end

run_bouncing_ball_example_all()

end # module BouncingBallExample
