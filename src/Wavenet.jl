# =====================================================================================
# - Data scaling to a given support interval
# - Construction of wavelet candidates (covering and k-means based)
# - Efficient, cache-friendly design-matrix building
# - Forward Stepwise Selection (Gram–Schmidt style) with threaded dot/projection
# - Backward Elimination with BLAS/LAPACK acceleration + minimal copying
# - SGD-style fine-tuning with safe CPU parallelization (thread-local reductions)
# - Safe usage of ConcurrentDict and immutable keys with NTuple for concurrency
# =====================================================================================

module Wavenet

using LinearAlgebra
using Random

using Base.Threads

using ConcurrentCollections: ConcurrentDict

export RadialActivation, WaveletCandidate,
       build_design_matrix, stepwise_wavelet_selection_wt,
       backward_elimination_wt, indices_to_candidates,
       finetune_from_candidates!

include("RadialActivation.jl")

# ----------------------- Scaling -----------------------
# Maps each row (dimension) of `data_points` from its original domain interval
# to a common target interval `support`. This is a simple per-dimension affine map:
# x_scaled = (x - dmin)/(dmax - dmin) * (hi - lo) + lo
# Edge case: if a domain has zero span, we map all its values to the center of support.
function data_scale_to_support(data_points::AbstractMatrix{<:Real},
                               domain::Vector{<:AbstractRange{<:Real}},
                               support::AbstractRange{<:Real})
    d, N = size(data_points)
    @assert d == length(domain) "rows(data_points) must match length(domain)."
    lo, hi = first(support), last(support)
    span = hi - lo
    @assert span != 0 "support must have non-zero length."

    scaled = Array{Float64}(undef, d, N)
    @inbounds for i in 1:d
        dmin, dmax = first(domain[i]), last(domain[i])
        domspan = dmax - dmin
        if domspan == 0
            scaled[i, :] .= lo + span/2          # map flat domain to center
        else
            @views scaled[i, :] = (data_points[i, :] .- dmin) ./ domspan .* span .+ lo
        end
    end
    return scaled
end

# ----------------------- Candidates -----------------------
# A candidate wavelet is identified by:
#  - w: positive scale (dilation)
#  - t: center (translation) vector in R^d
struct WaveletCandidate{T<:Real}
    w::T           # scale (dilation), scalar > 0
    t::Vector{T}   # center (translation), length d
end


# Generate a set of wavelet candidates that "cover" the scaled data.
# For each point x and dilatation level n, we compute integer lattice boxes m such that
# (x - w*m)/w lies within the (scaled) support; equivalent to x ∈ w[m + support].
# We store unique (n, m) indices in a concurrent set, then map them to (w, t=w*m).
function find_covering_wavelets(
    data_points_scaled::Matrix{<:Real},
    support::AbstractRange{<:Real},
    dilatation::UnitRange{<:Integer} = -2:4,
    supportScale::Real = 1.0
)::Vector{WaveletCandidate{Float64}}

    @assert maximum(data_points_scaled) <= last(support) &&
            minimum(data_points_scaled) >= first(support) "Data points must be within the support range."

    d, N = size(data_points_scaled)
    support_start, support_end = first(support)*supportScale, last(support)*supportScale

    # immutable, type-stable key: (n, m_tuple) where m_tuple::NTuple{d,Int}
    # Using NTuple avoids mutability hazards and ensures efficient hashing in Dict.
    wavelet_indices_set = ConcurrentDict{Tuple{Int, NTuple{d,Int}}, Nothing}()

    @threads for k in 1:N
        x = data_points_scaled[:, k]
        for n in dilatation
            scale = 2.0^n
            m_ranges = Vector{UnitRange{Int}}(undef, d)
            valid = true
            for i in 1:d
                m_lower = x[i]/scale - support_end
                m_upper = x[i]/scale - support_start
                m_start = ceil(Int, m_lower)
                m_end   = floor(Int, m_upper)
                if m_start > m_end
                    valid = false
                    break
                end
                m_ranges[i] = m_start:m_end
            end
            valid || continue

            # Iterators.product returns a Tuple{Int,...} of length d, matching NTuple{d,Int}
            for m_vec in Iterators.product(m_ranges...)
                wavelet_indices_set[(n, m_vec)] = nothing
            end
        end
    end

    # m is a tuple → collect to Vector{Int} for t = w .* m
    indices = [(n, collect(m)) for (n, m) in keys(wavelet_indices_set)]
    sort!(indices; by = x -> (x[1], Tuple(x[2])...))

    cands = WaveletCandidate{Float64}[]
    for (n, m) in indices
        w = 2.0^n
        t = w .* collect(Float64, m)
        push!(cands, WaveletCandidate(w, t))
    end
    return cands
end


using Clustering, Statistics


# Not working goodly needs improvement
# K-means-based proposal of wavelet centers (cluster centers) and scales (cluster radii).
# You can choose the statistic used for the cluster "radius" (mean, rms, p90, max) and
# multiply by `gain`. Scales are clamped to [min_w, max_w] for numerical safety.
function find_kmeans_wavelets(
    X::Matrix{Float64},                # d × N (columns = samples)
    K::Int;
    radius_stat::Symbol = :mean,       # :mean | :rms | :p90 | :max
    gain::Float64 = 1.0,               # scale multiplier
    min_w::Float64 = 1e-3,             # lower bound on scale
    max_w::Float64 = Inf,              # optional upper bound
    init::Symbol = :kmpp,              # k-means++ init
    maxiter::Int = 200,
    tol::Float64 = 1e-6,
    display::Symbol = :none
)::Vector{WaveletCandidate{Float64}}

    d, N = size(X)
    K = clamp(K, 1, N)

    R = kmeans(X, K; init=init, maxiter=maxiter, tol=tol, display=display)
    centers = R.centers            # d × K
    assign  = R.assignments        # length N

    cands = Vector{WaveletCandidate{Float64}}()
    for k in 1:K
        members = findall(==(k), assign)
        isempty(members) && continue

        dists = [norm(X[:, j] .- centers[:, k]) for j in members]

        r = if radius_stat === :mean
            mean(dists)
        elseif radius_stat === :rms
            sqrt(mean(abs2, dists))
        elseif radius_stat === :p90
            quantile(dists, 0.90)
        elseif radius_stat === :max
            maximum(dists)
        else
            error("Unknown radius_stat = $radius_stat")
        end

        w = clamp(gain * max(r, eps(Float64)), min_w, max_w)
        t = Vector{Float64}(@view centers[:, k])
        push!(cands, WaveletCandidate(w, t))   # (w, t) order!
    end

    isempty(cands) && error("find_kmeans_wavelets: all clusters empty; reduce K or check data.")
    return cands
end




# Optional adapter: (n,m) → WaveletCandidate
# Each tuple is (n::Int, m::Vector{Int}); effective t = w * m with w = 2^n
# This is convenient when you already have lattice indices and want actual (w,t).
function indices_to_candidates(Ws::Vector{Tuple{Int,Vector{Int}}})::Vector{WaveletCandidate{Float64}}
    M = length(Ws)
    M == 0 && return WaveletCandidate{Float64}[]
    d = length(Ws[1][2])
    cands = Vector{WaveletCandidate{Float64}}(undef, M)
    @inbounds for j in 1:M
        n, m = Ws[j]
        wj = 2.0^n
        tj = wj .* collect(Float64, m)
        @assert length(tj) == d
        cands[j] = WaveletCandidate{Float64}(wj, tj)
    end
    return cands
end

# ----------------------- Design Matrix -----------------------
# Build Ψ (N × L), columns are activations of candidates on all samples
# Ψ[k,j] = ψ( (X[:,k] - t_j)/w_j ).
# This is the “feature map” into wavelet activations; used by both selection methods.
function build_design_matrix(activation_function,
                             X::AbstractMatrix{<:Real},   # d × N
                             candidates::AbstractVector{<:WaveletCandidate})
    d, N = size(X)
    L = length(candidates)
    Ψ = Matrix{Float64}(undef, N, L)
    @inbounds for j in 1:L
        cj = candidates[j]
        wj = max(cj.w, 1e-9)
        tj = cj.t
        @assert length(tj) == d
        for k in 1:N
            @views Ψ[k, j] = activation_function.f((X[:, k] .- tj) ./ wj)
        end
    end
    return Ψ
end

# ----------------------- Stepwise Selection (forward) -----------------------
"""
    stepwise_wavelet_selection_wt(activation_function, candidates, X, y, s)

Forward stepwise selection over generic candidates (w,t).

Returns:
- α :: Vector{Float64}     (weights for selected)
- sel :: Vector{Int}       (indices into `candidates`)
- A, Q :: factorization matrices (as in your original routine)
"""
# Didactic notes:
# - Maintains a residualized copy `p` of the design matrix columns.
# - Each iteration picks the column with max correlation score ( (p_j'y)^2 / ‖p_j‖^2 ),
#   normalizes it to produce Q[:,i], then orthogonalizes remaining columns:
#   p_j ← p_j − (p_j·q_i) q_i.
# - The dot products and projections are threaded per column with @threads.
function stepwise_wavelet_selection_wt(activation_function, candidates, X, y, s)
    N = size(X, 2)
    Ψ = build_design_matrix(activation_function, X, candidates)  # N×L

    I = collect(1:size(Ψ,2))
    p = copy(Ψ)
    l = zeros(Int, s)
    Q = zeros(Float64, N, s)
    A = zeros(Float64, s, s)

    for i in 1:s
        numer = zeros(Float64, length(I))
        denom = zeros(Float64, length(I))

        @threads for t in eachindex(I)
            j = I[t]
            pj = @view p[:, j]
            numer[t] = dot(pj, y)^2
            denom[t] = dot(pj, pj)
        end
        scores = map((a,b)-> b>0 ? a/b : -Inf, numer, denom)
        best_local = argmax(scores)
        if scores[best_local] == -Inf
            l = l[1:i-1]; Q = Q[:, 1:i-1]; A = A[1:i-1, 1:i-1]
            break
        end

        l[i] = I[best_local]
        A[i,i] = norm(@view p[:, l[i]])
        if A[i,i] == 0
            l = l[1:i-1]; Q = Q[:, 1:i-1]; A = A[1:i-1, 1:i-1]
            break
        end
        @views Q[:, i] .= p[:, l[i]] ./ A[i,i]

        qi = @view Q[:, i]
        # Proietta in parallelo tutte le altre colonne
        @threads for t in eachindex(I)
            j = I[t]
            if j != l[i]
                pj = @view p[:, j]
                c  = dot(pj, qi)
                BLAS.axpy!(-c, qi, pj)
            end
        end

        setdiff!(I, l[i])  # togli la colonna selezionata
    end

    # risolvi i pesi nei selezionati
    wts = A \ (Q' * y)
    return wts, l, A, Q
end

# ----------------------- Backward Elimination -----------------------
"""
    backward_elimination_wt(activation_function, candidates, X, y, s)

Start from all candidates, greedily remove the one with smallest weight magnitude
until `s` remain. Uses ridge-stabilized normal equations.
"""
# Didactic notes:
# - Precompute Gram = Ψ'Ψ and ψy = Ψ'y.
# - Maintain an active index set `sel`, and in each step:
#   * assemble the k×k submatrix Gram[sel,sel] into a workspace,
#   * add λI, Cholesky-solve for current weights,
#   * drop the index with smallest |w|,
#   * shrink `sel` via swap+pop! (O(1)).
# - Parallelization: copy and argmin are threaded; the factorizations use BLAS/LAPACK threads.
function backward_elimination_wt(
    activation_function,
    candidates::AbstractVector,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    s::Integer;
    λ::Float64 = 1e-8
)
    N = size(X, 2)
    L = length(candidates)
    @assert N == length(y)
    @assert s < L "Final model size must be smaller than candidate set."

    # Build design matrix once (N×L)
    Ψ = build_design_matrix(activation_function, X, candidates)

    # Heavy lifting to BLAS (multithreaded gemm/gemv)
    gram = Matrix{Float64}(undef, L, L)
    mul!(gram, transpose(Ψ), Ψ)              # gram = Ψ' * Ψ
    ψy = Vector{Float64}(undef, L)
    mul!(ψy, transpose(Ψ), collect(Float64, y))  # ψy = Ψ' * y

    # Active set and workspaces
    sel = collect(1:L)                       # active indices
    nremove = L - s

    gwork = Matrix{Float64}(undef, L, L)     # strided workspace for sub-Gram
    rhs   = Vector{Float64}(undef, L)        # rhs = ψy[sel]
    wcur  = Vector{Float64}(undef, L)        # solution on current active set

    # To avoid BLAS oversubscription inside our threads, we only thread the copies and argmin.
    # Keep BLAS multithreaded for cholesky/solve *outside* of @threads regions.

    for _ in 1:nremove
        k = length(sel)                      # current active size

        # Copy gram[sel,sel] -> gwork[1:k,1:k]  (thread row-wise)
        @threads for ii in 1:k
            si = sel[ii]
            @inbounds for jj in 1:k
                gwork[ii, jj] = gram[si, sel[jj]]
            end
        end

        # rhs = ψy[sel]
        @inbounds @simd for ii in 1:k
            rhs[ii] = ψy[sel[ii]]
        end

        # Regularization on the diagonal
        @inbounds for ii in 1:k
            gwork[ii, ii] += λ
        end

        # Solve (gwork[1:k,1:k]) * wcur[1:k] = rhs[1:k]
        # Cholesky is very fast and multithreaded in good BLAS/LAPACK builds
        F = cholesky!(view(gwork, 1:k, 1:k); check=false)
        wv = view(wcur, 1:k)
        wv .= rhs[1:k]
        ldiv!(F, wv)                         # wv = F \ rhs

        # Drop the index with smallest |w|
        # Threaded reduction for argmin over 1:k
        best_pos = 1
        best_val = abs(@inbounds wcur[1])
        @threads for t in 2:k
            v = abs(@inbounds wcur[t])
            if v < best_val
                best_val = v
                best_pos = t
            end
        end

        # O(1) remove: swap with end then pop!
        sel[best_pos], sel[k] = sel[k], sel[best_pos]
        pop!(sel)
    end

    # Final weights on the remaining set
    k = length(sel)
    @threads for ii in 1:k
        si = sel[ii]
        @inbounds for jj in 1:k
            gwork[ii, jj] = gram[si, sel[jj]]
        end
    end
    @inbounds @simd for ii in 1:k
        rhs[ii] = ψy[sel[ii]]
        gwork[ii, ii] += λ
    end
    F = cholesky!(view(gwork, 1:k, 1:k); check=false)
    α = F \ view(rhs, 1:k)

    return α, sel
end


# Convenience wrapper that converts `candidates` → parameter arrays and calls
# the actual fine-tuning routine. It keeps your public API stable.
function finetune_from_candidates!(activation_function,
                                   candidates::AbstractVector{<:WaveletCandidate},
                                   α_init::AbstractVector{<:Real},
                                   X::AbstractMatrix{<:Real},
                                   y::AbstractVector{<:Real};
                                   epochs::Int=300, batch::Int=64, lr::Real=1e-2,
                                   l2::Real=1e-4, w_min::Real=1e-6,
                                   val_data::Union{Nothing,Tuple{Matrix{<:Real},Vector{<:Real}}}=nothing,
                                   rng=Random.default_rng())
    M = length(candidates)
    @assert M > 0 "No candidates provided."
    d = length(candidates[1].t)
    @assert size(X, 1) == d "Dim mismatch between candidates and X."

    w = Vector{Float64}(undef, M)
    t = Matrix{Float64}(undef, d, M)
    @inbounds for j in 1:M
        w[j] = candidates[j].w
        t[:, j] = candidates[j].t
    end
    α = collect(Float64, α_init)
    return finetune_wavenet!(activation_function, α, candidates, X, y;
                         epochs=epochs, batch=batch, lr=lr, l2=l2, w_min=w_min,
                         val_data=val_data, rng=rng)
end

# Mini-batch SGD fine-tuning with optional validation set.
# Uses a threaded, race-free batch kernel (`_sgd_batch!`) that:
# - splits the batch over threads,
# - accumulates gradients in thread-local buffers,
# - reduces them once, and
# - performs the parameter update serially.
function finetune_wavenet!(
    activation_function::RadialActivation.ActivationFunction,
    α::Vector{<:Real},                    # output weights (trainable)
    candidates::AbstractVector{<:WaveletCandidate}, # wavelet params (trainable w,t)
    X::Matrix{<:Real},                    # input data (d × N)
    y::Vector{<:Real};                    # targets
    epochs::Int=300,
    batch::Int=64,
    lr::Real=1e-2,
    l2::Real=1e-4,
    w_min::Real=1e-6,
    val_data::Union{Nothing,Tuple{Matrix{<:Real},Vector{<:Real}}}=nothing,
    rng=Random.default_rng()
)
    N = size(X,2)
    batch = min(batch, N)
    train_loss = Vector{Float64}(undef, epochs)
    val_loss   = isnothing(val_data) ? Float64[] : Vector{Float64}(undef, epochs)

    for e in 1:epochs
        idx = rand(rng, 1:N, batch)
        train_loss[e] = _sgd_batch!(activation_function, α, candidates, X, y, idx, lr, l2, w_min)

        if val_data !== nothing
            Xv, yv = val_data
            Nv = size(Xv,2)
            idxv = 1:Nv
            # eval-only → use copies so we don’t update params
            _α = copy(α)
            _cands = deepcopy(candidates)
            val_loss[e] = _sgd_batch!(activation_function, _α, _cands, Xv, yv, idxv, 0.0, 0.0, w_min)
        end
    end
    return α, candidates, train_loss, val_loss
end


# Threaded, allocation-light batch kernel (safe parallel CPU implementation).
# Key ideas:
# - Thread over samples; avoid locks by using per-thread gradient buffers.
# - Recompute z and its norm for grad_w/grad_t to avoid storing large intermediates.
# - SIMD in innermost loops; only one serial update of parameters at the end.
function _sgd_batch!(
    activation_function::RadialActivation.ActivationFunction,
    α::Vector{<:Real},
    candidates::AbstractVector{<:WaveletCandidate},
    X::Matrix{<:Real},
    y::Vector{<:Real},
    idx::AbstractVector{<:Integer},
    lr::Real, l2::Real, w_min::Real
)
    d, _ = size(X)
    M = length(candidates)
    B = length(idx)

    nt = nthreads()

    # Thread-local accumulators (avoid atomic adds)
    gradα_tls = [zeros(Float64, M) for _ in 1:nt]
    gradw_tls = [zeros(Float64, M) for _ in 1:nt]
    gradt_tls = [zeros(Float64, d, M) for _ in 1:nt]
    loss_tls  = zeros(Float64, nt)

    # Thread-local temporaries (reused per-sample to avoid allocations)
    psi_tls   = [zeros(Float64, M) for _ in 1:nt]
    zbuf_tls  = [Vector{Float64}(undef, d) for _ in 1:nt]

    @threads for t in 1:B
        tid = threadid()
        k   = idx[t]
        xk  = @view X[:, k]

        psi  = psi_tls[tid]
        zbuf = zbuf_tls[tid]

        # ---------- forward ----------
        yhat = 0.0
        @inbounds for j in 1:M
            wj = max(candidates[j].w, w_min)
            tj = candidates[j].t

            # fill zbuf = (xk - tj) / wj   (no allocs)
            @inbounds @simd for i in 1:d
                zbuf[i] = (xk[i] - tj[i]) / wj
            end

            ψ = activation_function.f(zbuf)
            psi[j] = ψ
            yhat += α[j] * ψ
        end

        ek = yhat - y[k]
        loss_tls[tid] += 0.5 * ek^2

        # ---------- backward ----------
        # grad_α: vectorized add
        @inbounds @simd for j in 1:M
            gradα_tls[tid][j] += ek * psi[j]
        end

        # grad_w, grad_t: recompute z and its norm (saves storing z/‖z‖ for all j)
        @inbounds for j in 1:M
            wj = max(candidates[j].w, w_min)
            tj = candidates[j].t

            # zbuf = (xk - tj)/wj  and accumulate norm^2
            s = 0.0
            @inbounds @simd for i in 1:d
                @inbounds zbuf[i] = (xk[i] - tj[i]) / wj
                s += zbuf[i]*zbuf[i]
            end
            nrm = sqrt(s)

            if nrm > 1e-9
                dψ = activation_function.df(nrm)
                common = ek * α[j] * dψ

                gradw_tls[tid][j] += common * (-nrm / wj)

                inv = -(common / (wj * nrm))
                @inbounds @simd for i in 1:d
                    gradt_tls[tid][i, j] += inv * zbuf[i]
                end
            end
        end
    end

    # ---------- reduce thread-local grads ----------
    grad_α = zeros(Float64, M)
    grad_w = zeros(Float64, M)
    grad_t = zeros(Float64, d, M)
    loss   = 0.0

    @inbounds for tid in 1:nt
        grad_α .+= gradα_tls[tid]
        grad_w .+= gradw_tls[tid]
        grad_t .+= gradt_tls[tid]
        loss   += loss_tls[tid]
    end

    # ---------- scale + regularize ----------
    grad_α ./= B
    grad_w ./= B
    grad_t ./= B

    if l2 > 0
        @. grad_α += l2 * α
        loss += 0.5 * l2 * sum(abs2, α)
    end

    loss /= B

    # ---------- parameter updates (serial, race-free) ----------
    @. α -= lr * grad_α
    @inbounds for j in 1:M
        candidates[j] = WaveletCandidate(
            max(candidates[j].w - lr * grad_w[j], w_min),
            candidates[j].t .- lr * @view(grad_t[:, j])
        )
    end

    return loss
end


# Stateless evaluation helper: computes ŷ = Σ_j w[j] * ψ((x - t_j)/w_j)
# for all samples (columns of X). Useful at inference time.
function wavenet_evaluate(
    activation_function,
    w::Vector{<:Real},        # output weights (length = M)
    wparams::Vector{<:Real},  # dilations (length = M)
    tparams::Matrix{<:Real},  # translations (d×M)
    X::Matrix{<:Real}         # input data (d×N)
)
    d, N = size(X)
    M = length(w)
    @assert length(wparams) == M "Length mismatch: wparams must match w"
    @assert size(tparams, 2) == M "Column count of tparams must match w"

    ŷ = zeros(Float64, N)

    @inbounds for k in 1:N
        s = 0.0
        xk = X[:, k]
        for j in 1:M
            s += w[j] * activation_function.f((xk .- tparams[:, j]) ./ max(wparams[j], 1e-6))
        end
        ŷ[k] = s
    end

    return ŷ
end


end # module Wavenet
