using Random
using MatrixEquations
using Flux

# ----------------------------------------------------
#
# LTI System
#
# ----------------------------------------------------

# LTI system
mutable struct lti
    A
    B
    C
    σw::AbstractFloat               # (To add Guassian white process noise)
    σv::AbstractFloat               # (To add Guassian white measurement noise)
    nx::Int64
    nu::Int64
    ny::Int64
    max_steps::Int64
    x0_lims::AbstractVector
end

function lti(
    A, B, C;
    σw = 0.0, σv = 0.0,
    max_steps=200, 
    x0_lims=[]
)
    nx, nu, ny = size(A,1), size(B, 2), size(C,1)
    isempty(x0_lims) && (G.x0_lims = ones(typeof(A[1,1]), nx))
    lti(A, B, C, σw, σv, nx, nu, ny, max_steps, x0_lims)
end

function linearised_cartpole(;dt=0.08, max_steps=50, σw=0.005, σv=0.001)

    δ, mp, l, mc, g = (dt, 0.2, 0.5, 1.0, 9.81)
    Ac = [0 1 0 0; 0 0 -mp*g/mc 0; 0 0 0 1; 0 0 g*(mc+mp)/(l*mc) 0]
    Bc = reshape([0; 1/mc; 0; -1/mc],4,1)
    Ag = Matrix(I,4,4)+δ*Ac
    Bg = δ*Bc
    Cg = [1.0 0 0 0; 0 0 1.0 0]
    x0_lims = [0.5, 0.2, 0.5, 0.2]/2

    G = lti(
        Ag, Bg, Cg; 
        x0_lims = x0_lims, 
        σw = σw, σv = σv, 
        max_steps = max_steps
    )

    return G

end

# Dynamics and measurement
(G::lti)(xt, ut, t=0) = G.A * xt + G.B * ut
measure(G::lti, xt, t=0) = G.C * xt

# Initialise system state
function init_state(G::lti, batches::Int, rng; zero_init=false)
    zero_init && (return zeros(G.nx,batches))
    return G.x0_lims .* randn(rng, G.nx, batches)
end

function init_state(G::lti, rng; zero_init=false)
    zero_init && (return zeros(G.nx))
    return G.x0_lims .* randn(rng, G.nx)
end

# Useful for generating Gaussian white noise
function noise_gen(G::lti, batches::Int, rng)
    wg = [ G.σw*randn(rng, G.nx, batches) for _ in 1:G.max_steps ]
    vg = [ G.σv*randn(rng, G.ny, batches) for _ in 1:G.max_steps ]
    return wg,vg
end


"""
For future systems, make a struct like lti with:
    - nx, nu, ny
    - max_steps
    - x0_lims
    - rng
Also required:
    - x0 = init_state(G, batches, x0_lims)
    - yt = measure(G, xt)
    - xt+1 = G(xt, ut) (one step of the dynamcis)
"""


# ----------------------------------------------------
#
# Output-feedback controller
#
# ----------------------------------------------------

# observer-based dynamic output feedback 
mutable struct ofb
    L   # observer update gain 
    K   # state feedback gain
end

# steady-state Kalman filter + LQR controller
function ofb(G::lti, Lb, Nb)

    nx = G.nx
    ny = G.ny
    nu = G.nu
    W = diagm(0 => Nb[1:nx]) #(G.σw^2)*Matrix(I,nx,nx) 
    V = diagm(0 => Nb[nx+1:nx+ny]) #(G.σv^2)*Matrix(I,ny,ny)
    Q = diagm(0 => Lb[1:nx]) 
    R = diagm(0 => Lb[nx+1:nx+nu])
    A = G.A
    B = G.B
    C = G.C
    S1 = zeros(nx,ny)
    X1, E1, F, Z1 = ared(A',C',V,W,S1) # F1'-> observer gain
    L = F'
    S2 = zeros(nx,nu)
    X2, E2, K, Z2 = ared(A,B,R,Q,S2)

    return ofb(L,K)

end

# closed-loop simulation for observer-based controller
function rollout(x0, Wg, Vg, G::lti, C::ofb)

    xh = 0*x0
    xe = (x0, xh)

    function f(x,t) # xe = [x, ξ]
        xg, xh = x
        yg = measure(G, xg) + Vg[t]
        yh = measure(G, xh)
        ωt = C.L*(yg-yh)
        ut = -C.K*xh
        xgn = G(xg,ut) + Wg[t]
        xhn = G(xh,ut) + ωt
        xn = (xgn, xhn)
        z = vcat(xg, ut)

        return xn, z
    end

    md = Flux.Recur(f,xe)
    z = md.(1:G.max_steps)

    return z

end

mutable struct lqg
    L
    K
end

# lqg design and rollout 
# see https://stanford.edu/class/ee363/lectures/lqg.pdf
function lqg(G::lti, Lo)
    L, K = [], []
    W = (G.σw^2)*Matrix(I, G.nx, G.nx) 
    V = (G.σv^2)*Matrix(I, G.ny, G.ny)
    X = diagm(0 => G.x0_lims.^2)
    Q = diagm(0 => Lo[1:G.nx]) 
    R = diagm(0 => Lo[G.nx+1:G.nx+G.nu])

    A, B, C, P = G.A, G.B, G.C, Q
    for _ in 1:G.max_steps

        Lt = (X*C') / (C*X*C' + V)
        L = push!(L, Lt)
        X = A*X*A' + W - A*Lt*C*X*A'

        Kt = (R + B'*P*B) \ (B'*P*A)
        K = push!(K, Kt)
        P = Q + A'*P*A - A'*P*B*Kt
    end

    return lqg(L, K)
end

function rollout(x0, Wg, Vg, G::lti, C::lqg)
    xe = (x0, 0*x0)
    T = G.max_steps

    function f(x,t) # xe = [x, ξ]
        xg, xp = x
        yg = measure(G,xg) + Vg[t]
        yh = measure(G,xp)
        xh = xp + C.L[t]*(yg-yh)
        ut = -C.K[T-t+1]*xh
        xpn = G(xh,ut) 
        xgn = G(xg,ut) + Wg[t]
        xn = (xgn, xpn)
        z = vcat(xg, ut)

        return xn, z
    end

    md = Flux.Recur(f,xe)
    z = md.(1:T)

    return z
end