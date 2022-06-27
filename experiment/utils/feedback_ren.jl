using Flux

includet("./LTI_utils.jl")
includet("./../../models/ffREN.jl")

mutable struct ren 
    param
end

Flux.trainable(m::ren) = Flux.trainable(m.param)

function ren(ny, nξ, nϕ, nu) 
    param = io_ren{Float64}(ny, nξ, nϕ, nu; polar_param=true)
    set_output_zero!(param)
    return ren(param)
end

function forward(Q::ren, Qe, x, u)
    xn, xw = Qe(x, u)
    v = Q.param.output(x,xw[2],u)
    return xn, v
end

function init_state(m::ren, batches)
    return zeros(m.param.nx, batches)
end

# closed-loop simulation for controller
function rollout(x0, Wg, Vg, G::lti, Q::ren, C::ofb)

    Qe = explicit(Q.param.implicit_cell)
    xh = 0*x0;
    xc = init_state(Q,size(x0,2))
    xe = (x0,xh,xc)

    function f(x,t) # x: closed-loop state
        xt, xh, xc = x
        yt = measure(G,xt) + Vg[t]
        yh = measure(G,xh)
        uq = yt - yh
        wt = C.L*uq
        xcn, v = forward(Q, Qe, xc, yt)
        ut = -C.K*xh + v
        xtn = G(xt,ut) + Wg[t]
        xhn = G(xh,ut) + wt
        xn = (xtn, xhn, xcn)
        z = vcat(xt, ut)

        return xn, z
    end

    md = Flux.Recur(f,xe)
    z = md.(1:G.max_steps)

    return z
end