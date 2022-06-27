cd(@__DIR__)
using Pkg
Pkg.activate("./..")

using Distributions
using LinearAlgebra
using Revise
using Flux
using Flux.Optimise:update!
using Flux.Optimise:ADAM
using Flux.Optimise:ClipValue
using Flux.Optimise:ExpDecay
using Zygote
using BSON
using Formatting
using CairoMakie
using StableRNGs

includet("./utils/LTI_utils.jl")
includet("./utils/feedback_ren.jl")

dir = "./results/"
name = "cp-uc-feedback-ren"

Random.seed!(0)

test_data = BSON.load(string(dir, "cp_uc.bson"))
# ------------------------------------------------------------
# keep this part unchanged across different LQ experiments 

# model definition
nx, nu, ny = (4, 1, 2)

Lb = test_data["Lb"] #10*rand(nx+nu) 
Lo = [1,1,5,1,1]
ub = 2
pb = 400
_u(u) = pb*max(abs(u) - ub, 0)
_cu(zt) = mean(_u.(zt[nx+nu,:]))
_cost(zt) = mean(sum(Lo .* zt.^2; dims=1))
cost(z::AbstractVector) = mean(_cost.(z)) + mean(_cu.(z))

G = linearised_cartpole(dt=0.08, max_steps=100)

function inject_dist!(G, wg, ub, rng)
    nb = size(wg[1],2)
    L = 5
    w = ub*(2*rand(rng, L, nb) .- 1) 
    t = rand(rng, L, nb) .+ 0.2
    td = zeros(Int, L, nb)
    ts = sum(t,dims=1)
    for b in 1:nb 
        td[:,b:b] = floor.(Int, G.max_steps/ts[b]*t[:,b:b])
        td[end, b] += G.max_steps - sum(td[:,b:b])
    end
    j = zeros(Int, 1, nb); tc = zeros(Int, 1, nb);
    wd = []
    for k in 1:G.max_steps
        wt = zeros(1,nb)
        for b in 1:nb
            if k > tc[b] 
                j[b] += 1
                tc[b] += td[j[b],b]
            end
            wt[b] = w[j[b], b]
        end
        wg[k] += G.B*wt
        wd = [wd..., wt]
    end
end

# generate test data
Batches = 100
X0 = test_data["x0"] #init_state(G, Batches, StableRNG(0))
_, Vg = noise_gen(G, Batches, StableRNG(0))
Wg = test_data["disturbance"]

# ------------------------------------------------------------
# backup controller
Nb = test_data["Nb"]
Cb = ofb(G, Lb, Nb)
zb = rollout(X0, Wg, Vg, G, Cb)

# cost function 50
Jb = cost(zb)

# hyper-parameter
nξ, nϕ, batches, Epoch, η, clip = (10, 20, 40, 500, 1e-3, 10.0)
step_decay_ep, step_decay_mag, step_decay_end = 0.8*Epoch, 0.1,  0.1

# youla controller setup
rng = StableRNG(0)
Q = ren(ny, nξ, nϕ, nu)
zt = rollout(X0, Wg, Vg, G, Q, Cb)
Jt = cost(zt)

# opt = Flux.Optimise.ADAM(η)
opt = Flux.Optimiser(
    ClipValue(clip), 
    ADAM(η),
    ExpDecay(1.0, step_decay_mag, step_decay_ep, step_decay_end)
    ) 
ps = Flux.Params(Flux.trainable(Q))

Jts = []
Tcs = []

for epoch in 1:Epoch

    # testing
    global zt = rollout(X0, Wg, Vg, G, Q, Cb)
    global Jt = cost(zt)

    # learning
    x0 = init_state(G, batches, rng)
    wg, vg = noise_gen(G, batches, rng)
    inject_dist!(G, wg, ub, rng)

    function loss()
        z = rollout(x0, wg, vg, G, Q, Cb)
        return cost(z)
    end

    t0 = time()
    J, back = Zygote.pullback(loss,ps)
    ∇J = back(one(J)) 
    update!(opt, ps, ∇J)   
    t1 = time()
    tc = t1-t0

    global Jts =[Jts..., Jt]
    global Tcs =[Tcs..., tc]

    printfmt("Epoch: {1:4d}, J_tr: {2:.2f}, J_tt: {3:.2f}, Jb: {4:.2f}, tc:{5:.2f}s\n", epoch, J, Jt, Jb, tc)

end

data = Dict(
    "Q"   => Q,
    "X0"  => X0,
    "Wg"  => Wg,
    "Vg"  => Vg,
    "Jts" => Jts,
    "Tcs" => Tcs,
    "Jb"  => Jb,
    "zb"  => zb,
    "zt"  => zt,
)

bson(string(dir, name, "-eta-",η, ".bson"),data)

# size_inches = (4, 2)
# size_pt = 100 .* size_inches
# f = Figure(resolution = size_pt, fontsize = 12)
# ax1 = Axis(f[1,1], xlabel = "Epochs", ylabel = "Cost")
# Epoch = length(Jts)
# xlims!(ax1, 0, Epoch)
# lines!(ax1, 1:Epoch, Jts*G.max_steps, color=:blue, label="Youla-REN")
# lines!(ax1, [1, Epoch], [Jb, Jb]*G.max_steps, color=:orange, label="Base-Linear")
# axislegend()
# display(f)

# save(string(dir,name,"-eta-",η, "-cost.pdf"),f,pt_per_unit = 1)
