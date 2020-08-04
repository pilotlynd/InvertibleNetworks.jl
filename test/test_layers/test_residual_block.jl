# Test residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, Test, Polynomials

# Input
nx = 32
ny = 32
n_in = 8
n_hidden = 16
batchsize = 2
k1 = 3
k2 = 3

# Input
X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# Weights
W1 = glorot_uniform(k1, k1, n_in, n_hidden)
W2 = glorot_uniform(k2, k2, n_hidden, n_hidden)
W3 = glorot_uniform(k1, k1, 2*n_in, n_hidden)
b1 = glorot_uniform(n_hidden)
b2 = glorot_uniform(n_hidden)

W01 = randn(Float32, size(W1))
W02 = randn(Float32, size(W2))
W03 = randn(Float32, size(W3))
b01 = randn(Float32, size(b1))
b02 = randn(Float32, size(b2))

dW1 = W1 - W01
dW2 = W2 - W02
dW3 = W3 - W03
db1 = b1 - b01
db2 = b2 - b02

# Residual blocks
RB = ResidualBlock(W1, W2, W3, b1, b2, nx, ny, batchsize)   # true weights

# Observed data
Y = RB.forward(X)

function loss(RB, X, Y; func=false)
    Y_ = RB.forward(X)
    ΔY = Y_ - Y
    f = .5f0*norm(ΔY)^2
    # Don't compute grads if only need loss value
    if func
        return f
    else
        ΔX = RB.backward(ΔY, X)
        return f, ΔX, RB.W1.grad, RB.W2.grad, RB.W3.grad, RB.b1.grad, RB.b2.grad
    end
end


# Gradient tests
maxiter = 5
h = collect([0.1f0/2^i for i=0:(maxiter-1)])

######################################  dX   ##################################################
# Gradient test w.r.t. input
f0, ΔX = loss(RB, X0, Y)[1:2]
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test convolutions dX\n")
for j=1:maxiter
    f = loss(RB, X0 + h[j]*dX, Y; func=true)
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h[j]*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
end

@show fit(log.(h), log.(err1), 1)
@show fit(log.(h), log.(err2), 1)

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


######################################  dW  ##################################################
# Gradient test for weights 1
RB0 = ResidualBlock(W01, W02, W03, b1, b2, nx, ny, batchsize)   # initial weights
f0, ΔX, ΔW1, ΔW2, ΔW3 = loss(RB0, X, Y)[1:5]
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions dW1 \n")
for j=1:maxiter
    RB0.W1.data = W01 + h[j]*dW1
    f = loss(RB0, X, Y; func=true)
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h[j]*dot(dW1, ΔW1))
    print(err3[j], "; ", err4[j], "\n")
end

@show fit(log.(h), log.(err3), 1)
@show fit(log.(h), log.(err4), 1)

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test for weights 2
RB0 = ResidualBlock(W01, W02, W03, b1, b2, nx, ny, batchsize)   # initial weights
f0, ΔX, ΔW1, ΔW2, ΔW3 = loss(RB0, X, Y)[1:5]
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions dW2 \n")
for j=1:maxiter
    RB0.W2.data = W02 + h[j]*dW2
    f = loss(RB0, X, Y; func=true)
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h[j]*dot(dW2, ΔW2))
    print(err3[j], "; ", err4[j], "\n")
end

@show fit(log.(h), log.(err3), 1)
@show fit(log.(h), log.(err4), 1)

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test for weights 3
RB0 = ResidualBlock(W01, W02, W03, b1, b2, nx, ny, batchsize)   # initial weights
f0, ΔX, ΔW1, ΔW2, ΔW3 = loss(RB0, X, Y)[1:5]
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions dW3 \n")
for j=1:maxiter
    RB0.W3.data = W03 + h[j]*dW3
    f = loss(RB0, X, Y; func=true)
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h[j]*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
end

@show fit(log.(h), log.(err3), 1)
@show fit(log.(h), log.(err4), 1)

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


######################################  db   ##################################################
# Gradient test for bias 1
RB0 = ResidualBlock(W1, W2, W3, b01, b02, nx, ny, batchsize)
f0, ΔX, ΔW1, ΔW2, ΔW3, Δb1, Δb2 = loss(RB0, X, Y)
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
print("\nGradient test convolutions dB1 \n")
for j=1:maxiter
    RB0.b1.data = b01 + h[j]*db1
    f = loss(RB0, X, Y; func=true)
    err5[j] = abs(f - f0)
    err6[j] = abs(f - f0 - h[j]*dot(db1, Δb1))
    print(err5[j], "; ", err6[j], "\n")
end

@show fit(log.(h), log.(err5), 1)
@show fit(log.(h), log.(err6), 1)

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for bias 2
RB0 = ResidualBlock(W1, W2, W3, b01, b02, nx, ny, batchsize)
f0, ΔX, ΔW1, ΔW2, ΔW3, Δb1, Δb2 = loss(RB0, X, Y)
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
print("\nGradient test convolutions dB2 \n")
for j=1:maxiter
    RB0.b2.data = b02 + h[j]*db2
    f = loss(RB0, X, Y; func=true)
    err5[j] = abs(f - f0)
    err6[j] = abs(f - f0 - h[j]*dot(db2, Δb2))
    print(err5[j], "; ", err6[j], "\n")
end

@show fit(log.(h), log.(err5), 1)
@show fit(log.(h), log.(err6), 1)

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)