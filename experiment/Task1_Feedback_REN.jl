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
name = "cp-lqg-feedback-ren"

Random.seed!(0)

test_data = BSON.load(string(dir, "cp_lqg.bson"))
# ------------------------------------------------------------
# keep this part unchanged across different LQ experiments 

# model definition
nx, nu, ny = (4, 1, 2)

Lb = test_data["Lb"] #10*rand(nx+nu) 
Lo = [1,1,5,1,1]
_cost(zt) = mean(sum(Lo .* zt.^2; dims=1))
cost(z::AbstractVector) = mean(_cost.(z))

G = linearised_cartpole(dt=0.08, max_steps=50)

# generate test data
Batches = 100
X0 = test_data["x0"] #init_state(G, Batches, StableRNG(0))
Wg, Vg = noise_gen(G, Batches, StableRNG(0))

# ------------------------------------------------------------
# backup controller
Nb = test_data["Nb"]
Cb = ofb(G, Lb, Nb)
zb = rollout(X0, Wg, Vg, G, Cb)

# LQG controller
Co = lqg(G, Lo)
zo = rollout(X0, Wg, Vg, G, Co)

# cost function 50
Jb = cost(zb)
Jo = cost(zo)

# hyper-parameter
nξ, nϕ, batches, Epoch, η, clip = (10, 20, 40, 200, 0.5*1e-2, 10.0)
step_decay_ep, step_decay_mag, step_decay_end = 0.8*Epoch, 0.1,  0.1

# youla controller setup
rng = StableRNG(0)

for n in 1:10
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
        zt = rollout(X0, Wg, Vg, G, Q, Cb)
        Jt = cost(zt)

        # learning
        x0 = init_state(G, batches, rng)
        wg, vg = noise_gen(G, batches, rng)

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

        Jts =[Jts..., Jt]
        Tcs =[Tcs..., tc]

        printfmt("Epoch: {1:4d}, J_tr: {2:.2f}, J_tt: {3:.2f}, Jb: {4:.2f}, Jo: {5:.2f}, tc:{6:.2f}s\n", epoch, J, Jt, Jb, Jo, tc)

    end

    data = Dict(
        "Q"   => Q,
        "X0"  => X0,
        "Wg"  => Wg,
        "Vg"  => Vg,
        "Jts" => Jts,
        "Tcs" => Tcs,
        "Jb"  => Jb,
        "Jo"  => Jo,
        "zb"  => zb,
        "zo"  => zo,
        "zt"  => zt,
    )

    bson(string(dir, name, "-eta-",η,"-n-",n, ".bson"),data)
end

# size_inches = (4, 2)
# size_pt = 100 .* size_inches
# f = Figure(resolution = size_pt, fontsize = 12)
# ax1 = Axis(f[1,1], xlabel = "Epochs", ylabel = "Cost")
# Epoch = length(Jts)
# xlims!(ax1, 0, Epoch)
# lines!(ax1, 1:Epoch, Jts*G.max_steps, color=:blue, label="Youla-REN")
# lines!(ax1, [1, Epoch], [Jb, Jb]*G.max_steps, color=:orange, label="Base-Linear")
# lines!(ax1, [1, Epoch], [Jo, Jo]*G.max_steps, linestyle=:dash, color=:black, label="LQG")
# axislegend()
# display(f)

# save(string(dir,name,"-eta-",η, "-cost.pdf"),f,pt_per_unit = 1)
