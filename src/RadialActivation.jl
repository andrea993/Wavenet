module RadialActivation



struct ActivationFunction
    f::Function   # Function: takes a vector z, returns a scalar
    df::Function  # Derivative: takes a SCALAR norm_z, returns a scalar dψ/d(||z||)
    support::AbstractRange{<:Real}
end

const MexicanHat = ActivationFunction(
    # f(z) = (d - ||z||²) * exp(-||z||²/2)
    z::Vector{<:Real} -> (length(z) - z' * z) * exp(-(z' * z) / 2),
    
    # df(norm_z) = norm_z * (norm_z² - 3) * exp(-norm_z²/2)
    # This function now takes a scalar norm and returns the scalar derivative.
    norm_z::Real -> let n2 = norm_z^2; norm_z * (n2 - 3) * exp(-n2 / 2); end,
    
    -5.0:5.0
)

const Gaussian = ActivationFunction(
    # f(z) = exp(-||z||²/2)
    z::Vector{<:Real} -> exp(-(z' * z) / 2),
    
    # df(norm_z) = -norm_z * exp(-norm_z²/2)
    norm_z::Real -> -norm_z * exp(-norm_z^2 / 2),
    
    -5.0:5.0
)



function Activation_nm(a::ActivationFunction, n::Integer, m::Vector{<:Integer})::Function
    scale = 2.0^n
    # The translation in space is m * scale (for a step size of 1)
    translation_vector = m * scale

    return function(x::Vector{<:Real})
        @assert length(x) == length(m) "Input vector x and translation index vector m must have the same dimension."
        
        # The argument to the mother wavelet is z = (x - t) / a
        z = (x .- translation_vector) / scale
        
        return a.f(z)
    end
end

# Convenience method for a tuple like Ws contains
# This is now (Int, Vector{Int})
function Activation_nm(a::ActivationFunction, nm::Tuple{Integer, Vector{<:Integer}})::Function
    return Activation_nm(a, nm[1], nm[2])
end

function Activation_wt(a::ActivationFunction, w::Real, t::Vector{<:Real})::Function
    return function(x::Vector{<:Real})
        @assert length(x) == length(t) "Input vector x and translation vector t must have the same dimension."
        
        # The argument to the mother wavelet is z = (x - t) / a
        z = (x .- t) / w
        
        return a.f(z)
    end
end

function dActivation_wt(a::ActivationFunction, w::Real, t::Vector{<:Real})::Function
    return function(x::Vector{<:Real})
        @assert length(x) == length(t) "Input vector x and translation vector t must have the same dimension."
        
        z = (x .- t) / w
        
        # Chain rule: ∇f(x) = (1/w) * ∇ψ(z)
        return a.df(z) / w
    end
end


end # module RadialActivation