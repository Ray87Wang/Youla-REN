using Flux

includet("./LTI_utils.jl")

mutable struct rnn
    A
    B1
    B2
    C1
    C2
    D1
    D2
    D3
    # bξ
    # bu
    # bv
    ϕ    # activation function
    lb   # activation slope lower bound
    ub   # activation slope upper bound
    nξ::Int64
    nϕ::Int64
    ny::Int64 
    nu::Int64
end

# Flux.trainable(C::rnn) = [C.A, C.B1, C.B2, C.C1, C.C2, C.D1, C.D2, C.D3, C.bξ, C.bu, C.bv]
Flux.trainable(C::rnn) = [C.A, C.B1, C.B2, C.C1, C.C2, C.D1, C.D2, C.D3]

function rnn(nξ, nϕ, ny, nu; ϕ=Flux.relu, lb=0.0, ub=1.0) 

    glorot_normal_matrix(n,m) = randn(n,m) / sqrt(n+m)
    #glorat_normal_vector(n)= randn(n) / sqrt(n)

    Θ  = glorot_normal_matrix(nξ+nu+nϕ, nξ+nϕ+ny)
    A  = Θ[1:nξ, 1:nξ]
    B1 = Θ[1:nξ, nξ+1:nξ+nϕ]
    B2 = Θ[1:nξ, nξ+nϕ+1:nξ+nϕ+ny]
    C1 = Θ[nξ+1:nξ+nu, 1:nξ]
    C2 = Θ[nξ+nu+1:nξ+nu+nϕ, 1:nξ]
    D1 = Θ[nξ+1:nξ+nu, nξ+1:nξ+nϕ]
    D2 = Θ[nξ+1:nξ+nu, nξ+nϕ+1:nξ+nϕ+ny]
    D3 = Θ[nξ+nu+1:nξ+nu+nϕ, nξ+nϕ+1:nξ+nϕ+ny]

    return rnn(A, B1, B2, C1, C2, D1, D2, D3, ϕ, lb, ub, nξ, nϕ, ny, nu)

    # bξ = glorat_normal_vector(nξ)
    # bu = glorat_normal_vector(nu)
    # bv = glorat_normal_vector(nϕ)

    # return rnn(A, B1, B2, C1, C2, D1, D2, D3, bξ, bu, bv, ϕ, lb, ub, nξ, nϕ, ny, nu)
end

init_state(C::rnn) = zeros(C.nξ)
init_state(C::rnn, batches) = zeros(C.nξ, batches)

function (C::rnn)(ξt,yt)
    vt = C.C2 * ξt + C.D3 * yt #.+ C.bv
    wt = C.ϕ.(vt)
    ut = C.C1 * ξt + C.D1 * wt + C.D2 * yt #.+ C.bu
    ξn = C.A * ξt + C.B1 * wt + C.B2 * yt #.+ C.bξ

    return ξn, ut
end

# CL system simulation
function rollout(x0, Wg, Vg, G::lti, C::rnn)

    ξ0 = init_state(C,size(x0,2))
    xe = (x0, ξ0)

    function f(x,t) # xe = [x, ξ]
        xt, ξt = x
        yt = measure(G, xt) + Vg[t]
        ξn, ut = C(ξt, yt)
        xn = G(xt,ut) + Wg[t]
        xen = (xn, ξn)
        z = vcat(xt, ut)

        return xen, z
    end

    md = Flux.Recur(f,xe)
    z = md.(1:G.max_steps)

    return z
end
