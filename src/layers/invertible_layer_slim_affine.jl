# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export AffineCouplingLayerSLIM


"""
    CS = AffineCouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=false, permute=false, k1=1, k2=3, p1=0, p2=1)

 Create an invertible SLIM coupling layer (affine).

 *Input*: 

 - `nx, ny`: spatial dimensions of input
 
 - `n_in`, `n_hidden`: number of input and hidden channels

 - `Ψ`: link function

 - `loget`: bool to indicate whether to return the logdet (default is `false`)

 - `permute`: bool to indicate whether to apply a channel permutation (default is `false`)

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 *Output*:
 
 - `CS`: Invertible SLIM coupling layer

 *Usage:*

 - Forward mode: `Y, logdet = CS.forward(X, D, A)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = CS.inverse(Y, D, A)`

 - Backward mode: `ΔX, X = CS.backward(ΔY, Y, D, A)`

 - where `A` is a linear forward modeling operator and `D` is the observed data.

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB` and 1x1 convolution layer `CL.C`

 See also: [`Conv1x1`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct AffineCouplingLayerSLIM <: NeuralNetLayer
    C::Union{Conv1x1, Nothing}
    RB::ResidualBlock
    Ψ::Function
    logdet::Bool
    forward::Function
    inverse::Function
    backward::Function
end


# Constructor from input dimensions
function AffineCouplingLayerSLIM(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64, Ψ::Function; 
    k1=4, k2=3, p1=0, p2=1, logdet::Bool=false, permute::Bool=false)

    # 1x1 Convolution and residual block for invertible layer
    permute == true ? (C = Conv1x1(n_in)) : (C = nothing)
    RB = ResidualBlock(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, fan=true)

    return AffineCouplingLayerSLIM(C, RB, Ψ, logdet,
        (X, D, J) -> forward_slim_affine(X, J, D, C, RB, Ψ; logdet=logdet),
        (Y, D, J) -> inverse_slim_affine(Y, J, D, C, RB, Ψ; logdet=logdet),
        (ΔY, Y, D, J) -> backward_slim_affine(ΔY, Y, J, D, C, RB, Ψ; logdet=logdet)
        )
end

# Forward pass: Input X, Output Y
function forward_slim_affine(X::Array{Float32, 4}, J, D, C, RB, Ψ; logdet=false)

    # Get dimensions
    nx, ny, n_s, batchsize = size(X)

    # Permute and split
    isnothing(C) ? (X_ = copy(X)) : (X_ = C.forward(X))
    X1_, X2_ = tensor_split(X_)

    # Gradient
    g = J'*(J*reshape(Ψ(X2_[:,:,1:1,:]), :, batchsize) - D)
    g = reshape(g/norm(g, Inf), nx, ny, 1, batchsize)
    X2_temp = tensor_cat(g, X2_[:,:,2:end,:])

    # Coupling layer
    Y2_ = copy(X2_)
    logS_T = RB.forward(X2_temp)
    logS, T = tensor_split(logS_T)
    S = Sigmoid(logS)
    Y1_ = S.*X1_ + T
    Y_ = tensor_cat(Y1_, Y2_)

    isnothing(C) ? (Y = copy(Y_)) : (Y = C.inverse(Y_))
    logdet == true ? (return Y, slim_logdet_forward(S)) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse_slim_affine(Y::Array{Float32, 4}, J, D, C, RB, Ψ; logdet=false, save=false)

    # Get dimensions
    nx, ny, n_s, batchsize = size(Y)
    isnothing(C) ?  (Y_ = copy(Y)) : (Y_ = C.forward(Y))

    # Gradient
    Y1_, Y2_ = tensor_split(Y_)
    X2_ = copy(Y2_)
    g = J'*(J*reshape(Ψ(X2_[:,:,1:1,:]), :, batchsize) - D)
    g = reshape(g/norm(g, Inf), nx, ny, 1, batchsize)
    X2_temp = tensor_cat(g, X2_[:,:,2:end,:])

    # Coupling layer
    logS_T = RB.forward(X2_temp)
    logS, T = tensor_split(logS_T)
    S = Sigmoid(logS)
    X1_ = (Y1_ - T) ./ (S + randn(Float32, size(S))*eps(1f0)) # add epsilon to avoid 
    X_ = tensor_cat(X1_, X2_)

    isnothing(C) ? (X = copy(X_)) : (X = C.inverse(X_))
    save == true ? (return X, X1_, X2_, X2_temp, S) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward_slim_affine(ΔY::Array{Float32, 4}, Y::Array{Float32, 4}, J, D, C, RB, Ψ; logdet=false, permute=false)

    # Recompute forward states
    X, X1_, X2_, X2_temp, S  = inverse_slim_affine(Y, J, D, C, RB, Ψ; logdet=logdet, save=true)

    # Backpropagation
    isnothing(C) ? (ΔY_ = copy(ΔY)) : (ΔY_ = C.forward((ΔY, Y))[1])
    ΔY1_, ΔY2_ = tensor_split(ΔY_)
    ΔT = copy(ΔY1_)
    ΔS = ΔY1_ .* X1_
    logdet == true && (ΔS -= slim_logdet_backward(S))
    ΔX1_ = ΔY1_ .* S
    ΔX2_ = RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), X2_temp) + ΔY2_
    ΔX_ = tensor_cat(ΔX1_, ΔX2_)
    isnothing(C) ? (ΔX = copy(ΔX_)) : (ΔX = C.inverse((ΔX_, tensor_cat(X1_, X2_)))[1])

    return ΔX, X
end

# Clear gradients
function clear_grad!(CS::AffineCouplingLayerSLIM)
    ~isnothing(CS.C) && clear_grad!(CS.C)
    clear_grad!(CS.RB)
end

# Get parameters
function get_params(CS::AffineCouplingLayerSLIM)
    p = get_params(CS.RB)
    ~isnothing(CS.C) && (p = cat(p, get_params(CS.C); dims=1))
    return p
end

# Logdet
slim_logdet_forward(S) = sum(log.(abs.(S))) / size(S, 4)
slim_logdet_backward(S) = 1f0./ S / size(S, 4)