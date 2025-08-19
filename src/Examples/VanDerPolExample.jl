module VanDerPolExample

include("../Wavenet.jl")

using .Wavenet
using Plots
using Random

# Domain helper (per-row min/max with padding), for X as d×N
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

# ------------------------ Data gen (Van der Pol) ------------------------

# 4th-order Runge-Kutta step
function rk4_step(f, u, t, dt, mu)
    k1 = f(u, t, mu)
    k2 = f(u .+ (dt/2) .* k1, t + dt/2, mu)
    k3 = f(u .+ (dt/2) .* k2, t + dt/2, mu)
    k4 = f(u .+ dt .* k3, t + dt, mu)
    return u .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function generate_vanderpol_data(n_samples::Int=1000, t_end::Real=30.0, mu::Real=1.5)
    function vanderpol(u, t, mu)
        x, y = u
        dx = y
        dy = mu * (1 - x^2) * y - x
        return [dx, dy]
    end
    dt = t_end / n_samples
    t  = 0.0
    u  = [1.0, 1.0]   # initial state
    X  = zeros(Float64, n_samples, 2)
    T  = zeros(Float64, n_samples)
    @inbounds for i in 1:n_samples
        X[i, :] = u
        T[i]    = t
        u = rk4_step(vanderpol, u, t, dt, mu)
        t += dt
    end
    return X, T
end

# ------------------------ Training / Prediction ------------------------

function train_vanderpol_wavenet(X::Matrix{<:Real}, y::Vector{<:Real};
    s::Int=40, epochs::Int=1000, batch::Int=64, lr::Real=0.2, l2::Real=0.0,
    pad::Real=0.5, dilatation=-2:6)

    @assert size(X,2) == length(y) "X and y length mismatch."
    Wavelet = Wavenet.RadialActivation.MexicanHat
    support = Wavelet.support

    # domain from full (train) X
    domain = _domain_from_data(X; pad=pad)

    # scale inputs to support
    Xs = Wavenet.data_scale_to_support(X, domain, support)

    # build lattice candidates (as WaveletCandidate)
    candidates_all = Wavenet.find_covering_wavelets(Xs, support, dilatation, 0.3)
    @assert !isempty(candidates_all) "No candidate wavelets from covering. Try larger pad or adjust dilatation."

    # forward stepwise selection over candidates
    #(w_init, sel_idx, A, Q) = Wavenet.stepwise_wavelet_selection_wt(Wavelet, candidates_all, Xs, y, s)
    (w_init, sel_idx) = Wavenet.backward_elimination_wt(
        Wavelet, candidates_all, Xs, y, s;)
    
    candidates_selected = candidates_all[sel_idx]

    # finetune from selected candidates (returns updated candidates and output weights)
    (w, cands_tr, tr_loss, _) = Wavenet.finetune_from_candidates!(
        Wavelet, candidates_selected, w_init, Xs, y;
        epochs=epochs, batch=batch, lr=lr, l2=l2
    )

    # extract learned (wparams, tparams)
    wparams = [c.w for c in cands_tr]
    tparams = stack_t(cands_tr)


    

    return Wavelet, w, wparams, tparams, candidates_selected, candidates_all, Xs, domain, support
end

function predict_vanderpol_wavenet(Wavelet, w::Vector{<:Real},
                                   wparams::Vector{<:Real}, tparams::Matrix{<:Real},
                                   X_test::Matrix{<:Real}, domain, support)
    Xs_test = Wavenet.data_scale_to_support(X_test, domain, support)
    return Wavenet.wavenet_evaluate(Wavelet, w, wparams, tparams, Xs_test)
end


function simulate_vanderpol_wavenet(model_params, initial_state::Vector{<:Real},
                                    n_steps::Int, domain, support)
    Wavelet, w, wparams, tparams = model_params
    @assert length(initial_state) == 2 "Initial state must contain two values."
    traj  = zeros(Float64, n_steps + 2)
    traj[1:2] = initial_state
    state = copy(initial_state)
    @inbounds for i in 1:n_steps
        Xcol = reshape(state, 2, 1)
        Xs   = Wavenet.data_scale_to_support(Xcol, domain, support)
        nxt  = Wavenet.wavenet_evaluate(Wavelet, w, wparams, tparams, Xs)[1]
        traj[i+2] = nxt
        state[1] = state[2]
        state[2] = nxt
    end
    return traj[3:end]
end

# ------------------------ End-to-end example ------------------------

function run_vanderpol_example(; n_samples=2000, t_end=300.0, mu=1.5,
                               s=100, epochs=2000, batch=64, lr=0.2, l2=0.0,
                               train_ratio=0.8, pad=0.5, dilatation=-2:6,
                               seed=123)
    Random.seed!(seed)
    Xraw, _t = generate_vanderpol_data(n_samples, t_end, mu)  # Xraw: N×2

    # Use only the first coordinate, build a 2-lag prediction task: x(t+2) from [x(t), x(t+1)]
    x = Xraw[:, 1]
    Ncols = n_samples - 2
    X_wavenet = Matrix{Float64}(undef, 2, Ncols)
    y_wavenet = Vector{Float64}(undef, Ncols)
    @inbounds for i in 1:Ncols
        X_wavenet[:, i] = x[i:i+1]
        y_wavenet[i]    = x[i+2]
    end

    # Split train/test by columns
    Ntrain = floor(Int, train_ratio * Ncols)
    train_idx = 1:Ntrain
    test_idx  = (Ntrain+1):Ncols
    X_train, y_train = X_wavenet[:, train_idx], y_wavenet[train_idx]
    X_test,  y_test  = X_wavenet[:, test_idx],  y_wavenet[test_idx]

    # Train
    Wavelet, w, wparams, tparams, Ws_sel, Ws_all, Xs_train, domain, support =
        train_vanderpol_wavenet(X_train, y_train;
            s=s, epochs=epochs, batch=batch, lr=lr, l2=l2, pad=pad, dilatation=dilatation)

    # 1-step test
    ŷ_test = predict_vanderpol_wavenet(Wavelet, w, wparams, tparams, X_test, domain, support)

    # Rollout on test
    init_state = X_test[:, 1]
    nsteps     = length(y_test)
    model_params = (Wavelet, w, wparams, tparams)
    y_rollout = simulate_vanderpol_wavenet(model_params, init_state, nsteps, domain, support)

    # Plot
    idx = collect(test_idx)
    p = plot(idx, y_test, label="True Trajectory (Test)", lw=2)
    plot!(p, idx, ŷ_test, label="1-step ahead", lw=2, ls=:dash)
    plot!(p, idx, y_rollout, label="Simulated Trajectory", lw=2, ls=:dot)
    title!(p, "Van der Pol — s=$s, epochs=$epochs")
    xlabel!("Time Step"); ylabel!("x(t)"); plot!(p, legend=:bottomright)
    display(p)
    return p
end

# auto-run
run_vanderpol_example()

end # module VanDerPolExample
