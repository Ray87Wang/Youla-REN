using Flux:LSTMCell
using Revise
using Distributions

includet("./LTI_utils.jl")

mutable struct lstm 
    nu
    nv
    ny
    A
    B
    C
    bx
    by
end

function lstm(nu, nv, ny)    
    nx = 4 * nv 
    A = randn(nx,nv)/sqrt(nv+nx)
    B = randn(nx,nu)/sqrt(nx+nu)
    C = randn(ny,nv)/sqrt(ny+nv)
    bx = zeros(nx)/sqrt(nx)
    by = zeros(ny)/sqrt(ny)
    bx[nv+1:2*nv] .= 1

    return lstm(nu, nv, ny, A, B, C, bx, by)
end

Flux.trainable(m::lstm) = (m.A, m.B, m.C, m.bx, m.by)

function (m::lstm)(x0, u)
    xt = m.A * x0[1:m.nv,:] + m.B * u .+ m.bx
    ft = Flux.sigmoid.(xt[1:m.nv,:])
    it = Flux.sigmoid.(xt[m.nv+1:2*m.nv,:])
    ot = Flux.sigmoid.(xt[2*m.nv+1:3*m.nv,:])
    ct = Flux.tanh.(xt[3*m.nv+1:4*m.nv,:])
    c  = ft .* x0[m.nv+1:2*m.nv,:] .+ it .* ct
    h  = ot .* Flux.tanh.(c)
    y  = m.C * h .+ m.by 

    return vcat(h,c), y 
end

init_state(m::lstm) = zeros(2*m.nv)
init_state(m::lstm, batches) = zeros(2*m.nv, batches)

# function set_output_zero!(m::lstm)
#     m.C  .*= 0.0
#     m.by .*= 0.0

#     return nothing
# end

# closed-loop simulation for youla controller
function rollout(x0, Wg, Vg, G::lti, Q::lstm, C::ofb)

    xh = 0*x0;
    xc = init_state(Q,size(x0,2))
    xe = (x0,xh,xc)

    function f(x,t) # x: closed-loop state
        xt, xh, xc = x
        yt = measure(G,xt) + Vg[t]
        yh = measure(G,xh)
        uq = yt - yh
        wt = C.L*uq
        xcn, v = Q(xc, uq)
        ut = -C.K*xh + v
        xtn = G(xt,ut) + Wg[t]
        xhn = G(xh,ut) + wt
        xn = (xtn, xhn, xcn)
        z = vcat(xtn, ut)

        return xn, z
    end

    md = Flux.Recur(f,xe)
    z = md.(1:G.max_steps)

    return z
end