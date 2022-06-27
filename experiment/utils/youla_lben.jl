using Flux

includet("./LTI_utils.jl")
includet("./../../models/ffREN.jl")

mutable struct lben
    param
end

Flux.trainable(m::lben) = Flux.trainable(m.param)

function lben(ny, nϕ, nu) 
    param = io_ren{Float64}(ny, 0, nϕ, nu; polar_param=true)
    set_output_zero!(param)
    return lben(param)
end

function forward(Q::lben, Qe, u)
    b = Qe.D12 * u .+ Qe.bv
    w = tril_eq_layer(Qe.ϕ, Qe.D11, b)
    return Q.param.output.D21 * w + Q.param.output.D22 * u .+ Q.param.output.by
end

# closed-loop simulation for youla controller
function rollout(x0, Wg, Vg, G::lti, Q::lben, C::ofb)

    Qe = explicit(Q.param.implicit_cell)
    xh = 0*x0;
    xe = (x0,xh)

    function f(x,t) # x: closed-loop state
        xt, xh = x
        yt = measure(G,xt) + Vg[t]
        yh = measure(G,xh)
        uq = yt - yh
        wt = C.L*uq
        v = forward(Q, Qe, uq)
        ut = -C.K*xh + v
        xtn = G(xt,ut) + Wg[t]
        xhn = G(xh,ut) + wt
        xn = (xtn, xhn)
        z = vcat(xt, ut)

        return xn, z
    end

    md = Flux.Recur(f,xe)
    z = md.(1:G.max_steps)

    return z
end