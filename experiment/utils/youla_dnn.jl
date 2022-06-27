using Flux

includet("./LTI_utils.jl")

mutable struct dnn
    W1
    W2
    W3
    b1
    b2
    b3
    ϕ
end

Flux.trainable(m::dnn) = [m.W1, m.W2, m.W3, m.b1, m.b2, m.b3]

function dnn(nu, nhidden, ny; ϕ=relu, initW = Flux.glorot_uniform(Random.GLOBAL_RNG)) 
    W1 = initW(nhidden, nu)
    W2 = initW(nhidden, nhidden)
    W3 = zeros(ny, nhidden)
    b1 = zeros(nhidden)
    b2 = zeros(nhidden)
    b3 = zeros(ny)

    return dnn(W1, W2, W3, b1, b2, b3, ϕ)
end

function forward(m::dnn, u)
    x1 = m.ϕ.(m.W1 * u .+ m.b1)
    x2 = m.ϕ.(m.W2 * x1 .+ m.b2)
    return m.W3 * x2 .+ m.b3
end

# closed-loop simulation for youla controller
function rollout(x0, Wg, Vg, G::lti, Q::dnn, C::ofb)

    xh = 0*x0;
    xe = (x0,xh)

    function f(x,t) # x: closed-loop state
        xt, xh = x
        yt = measure(G,xt) + Vg[t]
        yh = measure(G,xh)
        uq = yt - yh
        wt = C.L*uq
        v = forward(Q, uq)
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