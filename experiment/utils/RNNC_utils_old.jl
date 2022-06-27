using Flux
using JuMP
using MosekTools
using MatrixEquations

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

function rnn_init(nξ, nϕ, ny, nu; ϕ=tanh, lb=0.0, ub=1.0) 

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
        z = vcat(xn, ut)

        return xen, z
    end

    md = Flux.Recur(f,xe)
    z = md.(1:G.max_steps)

    return z
end

# LMI projection
function solve_lmi(G::lti, C::rnn, P, Λ, ρ)

    s  = (C.ub + C.lb)/2
    t  = (C.ub - C.lb)/2
    nξ = C.nξ
    nϕ = C.nϕ
    ny = C.ny
    nu = C.nu
    nx = G.nx

    Atk  = C.A + s*C.B1 * C.C2
    Btk1 = t*C.B1
    Btk2 = C.B2 + s*C.B1 * C.D3
    Ctk1 = C.C1 + s*C.D1 * C.C2
    Dtk1 = t*C.D1
    Dtk2 = C.D2 + s*C.D1 * C.D3
    Ctk2 = C.C2
    Dtk3 = C.D3
    Θt = [Atk Btk1 Btk2; Ctk1 Dtk1 Dtk2; Ctk2 zeros(nϕ,nϕ) Dtk3]
    P1 = inv(P)
    Λ1 = inv(Λ)

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true, "INTPNT_CO_TOL_DFEAS" => 1e-7))

    @variables(model, begin
        Ak[1:nξ, 1:nξ]  
        Bk1[1:nξ, 1:nϕ] 
        Bk2[1:nξ, 1:ny]    
        Ck1[1:nu, 1:nξ]  
        Ck2[1:nϕ, 1:nξ]
        Dk1[1:nu, 1:nϕ]  
        Dk2[1:nu, 1:ny]  
        Dk3[1:nϕ, 1:ny]
    end)
    @variable(model, Q1[1:nξ+nx, 1:nξ+nx], PSD)
    @variable(model, q2[1:nϕ] >= 1e-5)

    Q2 = diagm(0 => q2)
    Ac = [G.A+G.B*Dk2*G.C G.B*Ck1; Bk2*G.C Ak]
    Bc = [G.B*Dk1; Bk1]
    Cc = [Dk3*G.C Ck2]
    Dc = zeros(nϕ, nϕ)
    Z = zeros(nξ+nx, nϕ)
    @SDconstraint(model, [ρ^2*(2*P-P'*Q1*P) Z Ac' Cc';
                          Z' 2*Λ-Λ'*Q2*Λ Bc' Dc';
                          Ac Bc Q1 Z;
                          Cc Dc Z' Q2] >= 0)
    
    Θ = [Ak Bk1 Bk2; Ck1 Dk1 Dk2; Ck2 zeros(nϕ,nϕ) Dk3]
    dΘ = Θ-Θt
    dQ1 = Q1-P1
    dQ2 = Q2-Λ1
    @objective(model, Min, tr(dΘ*dΘ') + tr(dQ1*dQ1') + tr(dQ2*dQ2'))

    optimize!(model)

    ts = termination_status(model)
    
    Θv = value.(Θ)
    Qv1 = value.(Q1)
    Qv2 = value.(Q2)
    # P = inv(Qv1)
    # Λ = inv(Qv2)

    return Qv1, Qv2, Θv, ts
end

function proj!(C::rnn, Θ)
    s  = (C.ub + C.lb)/2
    t  = (C.ub - C.lb)/2
    nξ = C.nξ
    nϕ = C.nϕ
    ny = C.ny
    nu = C.nu

    Avk  = Θ[1:nξ, 1:nξ]
    Bvk1 = Θ[1:nξ, nξ+1:nξ+nϕ]
    Bvk2 = Θ[1:nξ, nξ+nϕ+1:nξ+nϕ+ny]
    Cvk1 = Θ[nξ+1:nξ+nu, 1:nξ]
    Cvk2 = Θ[nξ+nu+1:nξ+nu+nϕ, 1:nξ]
    Dvk1 = Θ[nξ+1:nξ+nu, nξ+1:nξ+nϕ]
    Dvk2 = Θ[nξ+1:nξ+nu, nξ+nϕ+1:nξ+nϕ+ny]
    Dvk3 = Θ[nξ+nu+1:nξ+nu+nϕ, nξ+nϕ+1:nξ+nϕ+ny]

    C.B1 = 1/t*Bvk1
    C.D1 = 1/t*Dvk1
    C.C2 = Cvk2
    C.D3 = Dvk3
    C.D2 = Dvk2 - s*C.D1 * C.D3
    C.C1 = Cvk1 - s*C.D1 * C.C2
    C.B2 = Bvk2 - s*C.B1 * C.D3
    C.A  = Avk - s*C.B1 * C.C2
end

function P_init(G::lti,F::ofb)

    nx = G.nx
    Ad = [G.A -G.B*F.K; F.L*G.C G.A-G.B*F.K-F.L*G.C]
    ρ = 0.99
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true, "INTPNT_CO_TOL_DFEAS" => 1e-7))

    @variables(model, begin 
        P[1:2*nx, 1:2*nx], PSD 
    end)

    H = [ρ^2*P Ad'*P; P*Ad P]
    @SDconstraint(model, H >= 0)
    @SDconstraint(model, P - 1E-3*Matrix(I,2*nx,2*nx) >= 0)
    optimize!(model)
    #termination_status(model)

    return value.(P), ρ
end