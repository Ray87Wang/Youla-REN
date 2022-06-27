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
includet("./utils/rnn_utils.jl")
includet("./utils/projector.jl")

dir = "./results/"
name = "cp-lqg-projection-rnn"

Random.seed!(0)

test_data = BSON.load(string(dir, "cp_lqg.bson"))

Random.seed!(0)

# ------------------------------------------------------------
# keep this part unchanged across different LQ experiments 

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
nξ, nϕ, batches, Epoch, η, clip = (10, 20, 40, 500, 1e-3, 10.0)
step_decay_ep, step_decay_mag, step_decay_end = 0.8*Epoch, 0.1,  0.1

rng = StableRNG(0)
# RNN controller init
Q = rnn(nξ,nϕ,ny,nu)
# projector init 
P = projector(G,Q)
# projection
# projection!(C,P)
zt = rollout(X0, Wg, Vg, G, Q)
Jt = cost(zt)

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
    global zt = rollout(X0, Wg, Vg, G, Q)
    global Jt = cost(zt)

    # learning
    x0 = init_state(G, batches, rng)
    wg, vg = noise_gen(G, batches, rng)

    function loss()
        z = rollout(x0, wg, vg, G, Q)
        return cost(z)
    end

    t0 = time()
    J = 0.0
    if epoch > 1 # skip the first step as the model might be unstable
        J, back = Zygote.pullback(loss,ps)
        ∇J = back(one(J)) 
        update!(opt, ps, ∇J)  
    end 

    projection!(Q, P)
    t1 = time()
    tc = t1-t0

    global Jts =[Jts..., Jt]
    global Tcs =[Tcs..., tc]

    if Jt > 1e5
        printfmt("Epoch: {1:4d} unstable model\n", epoch)
    else
        printfmt("Epoch: {1:4d}, J_tr: {2:.2f}, J_tt: {3:.2f}, Jb: {4:.2f}, Jo: {5:.2f}, tc:{6:.2f}s\n", epoch, J, Jt, Jb, Jo, tc)
    end

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

bson(string(dir, name, "-eta-",η, ".bson"),data)

size_inches = (4, 2)
size_pt = 100 .* size_inches
f = Figure(resolution = size_pt, fontsize = 12)
ax1 = Axis(f[1,1], xlabel = "Epochs", ylabel = "Cost")
Epoch = length(Jts)
Jts[1] = Jts[2] + 40
xlims!(ax1, 0, Epoch)
ylims!(ax1, 10, 40)
lines!(ax1, 1:Epoch, Jts, color=:blue, label="Projection")
lines!(ax1, [1, Epoch], [Jb, Jb], color=:orange, label="Base-Linear")
lines!(ax1, [1, Epoch], [Jo, Jo], linestyle=:dash, color=:black, label="LQG")
axislegend()
display(f)

save(string(dir,name,"-eta-",η, "-cost.pdf"),f,pt_per_unit = 1)
