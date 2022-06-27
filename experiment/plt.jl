cd(@__DIR__)
using Pkg
Pkg.activate("./..")

using Distributions
using LinearAlgebra
using Revise
using Flux
using BSON
using CairoMakie
using StableRNGs

includet("./utils/LTI_utils.jl")
includet("./utils/youla_ren.jl")
includet("./utils/youla_lben.jl")
includet("./utils/youla_dnn.jl")
includet("./utils/youla_lstm.jl")
includet("./utils/feedback_ren.jl")
includet("./utils/feedback_lben.jl")
includet("./utils/feedback_dnn.jl")
includet("./utils/feedback_lstm.jl")
includet("./utils/rnn_utils.jl")

dir = "./results/"
lqg_files = [
    "cp-lqg-projection-rnn-eta-0.01",
    "cp-lqg-feedback-dnn-eta-0.01",
    "cp-lqg-feedback-lben-eta-0.01",
    "cp-lqg-feedback-lstm-eta-0.1",
    "cp-lqg-feedback-ren-eta-0.005",
    "cp-lqg-youla-dnn-eta-0.01",
    "cp-lqg-youla-lben-eta-0.01",
    "cp-lqg-youla-lstm-eta-0.1",
    "cp-lqg-youla-ren-eta-0.01"
]

data = BSON.load(string(dir,lqg_files[9],"-n-", 1, ".bson"))
Jb = data["Jb"]
Jo = data["Jo"]

models = [
    "Project-RNN",
    "Feedback-DNN",
    "Feedback-LBEN",
    "Feedback-LSTM",
    "Feedback-REN",
    "Youla-DNN",
    "Youla-LBEN",
    "Youla-LSTM",
    "Youla-REN"
]

colours = [
    :red,
    :blue, :green, :grey, :purple,
    :blue, :green, :grey, :purple
]

lstyle = [
    :solid,
    :dash, :dash, :dash, :dash,
    :solid, :solid, :solid, :solid
]

size_inches = (7, 4)
size_pt = 100 .* size_inches
Epoch = 200
epoch = [i for i in 1:Epoch]

f = Figure(resolution = size_pt, fontsize = 18)
ga = f[1,1] = GridLayout()
ax = Axis(ga[1,1], xlabel = "Epochs", ylabel = "Normalized cost")

for k in 1:length(lqg_files)
    J = 1/Jb*reduce(hcat,[BSON.load(string(dir,lqg_files[k],"-n-", n, ".bson"))["Jts"] for n in 1:10])
    Jm = vec(minimum(J;dims=2))
    JM = vec(maximum(J;dims=2))
    Ja = vec(mean(J; dims=2))
    Js = vec(std(J; dims=2))
    
    if k == 1
        lines!(ax, [1, Epoch], [1, 1], color=:black, linestyle=:solid, linewidth=2, label="Base-Control")
        Jm[1]=2*Jb
        Js[1]=0.0
    end
    
    #band!(ax, epoch, Ja + Js, Ja - Js, color = (colours[k], 0.3))
    band!(ax, epoch, Jm, JM, color = (colours[k], 0.3))
    lines!(ax, Ja, label= models[k], linewidth=1.5, color=colours[k], linestyle=lstyle[k])

    if k == length(lqg_files)
        lines!(ax, [1, Epoch], [Jo, Jo]/Jb, color=:black, linestyle=:dashdot, linewidth=2, label="Optimal")
    end
end

xlims!(ax, (0,Epoch))
ylims!(ax, (0,1.5))
Legend(ga[1,2], ax,orientation = :vertical)
display(f)
save(string(dir,"cp-lqg.pdf"),f)

# files = [
#     "cp-uc-projection-rnn-eta-0.001.bson",
#     "cp-uc-feedback-dnn-eta-0.01.bson",
#     "cp-uc-feedback-lben-eta-0.001.bson",
#     "cp-uc-feedback-lstm-eta-0.1.bson",
#     "cp-uc-feedback-ren-eta-0.001.bson",
#     "cp-uc-youla-dnn-eta-0.01.bson",
#     "cp-uc-youla-lben-eta-0.01.bson",
#     "cp-uc-youla-lstm-eta-0.1.bson",
#     "cp-uc-youla-ren-eta-0.01.bson"
# ]

# size_inches = (7, 4)
# size_pt = 100 .* size_inches
# f = Figure(resolution = size_pt, fontsize = 18)
# ga = f[1,1] = GridLayout()
# ax = Axis(ga[1,1], xlabel = "Epochs", ylabel = "Normalized cost")
# Epoch = 500

# for k in 1:length(files)
#     data = BSON.load(string(dir, files[k]))
#     J = data["Jts"]
#     Jb = data["Jb"]

#     if k == 1
#         lines!(ax, [1, Epoch], [1, 1], color=:black, linestyle=:solid, linewidth=2, label="Base-Control")
#         J[1]=1.4*Jb
#     end
    
#     lines!(ax, J/Jb, label=models[k], color=colours[k], linestyle=lstyle[k])

# end

# xlims!(ax, (0,Epoch))
# ylims!(ax, (0,1.2))
# Legend(ga[1,2], ax,orientation = :vertical)
# display(f)
# save(string(dir,"cp-uc.pdf"),f)