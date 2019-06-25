cd(@__DIR__)
using Plots, Flux, Zygote, LinearAlgebra, Parameters, Statistics, Random, Printf, OrdinaryDiffEq, DSP, SparseArrays, Dates
using Flux: params
using Serialization: serialize, deserialize
default(lab="", grid=false)
# include("SkipRNN.jl")

@userplot Eigvalplot
@recipe function eigvalplot(A::Eigvalplot)
    e = data(A.args[1]) |> eigvals# .|> log
    title --> "Eigenvalues"
    ar = range(0, stop=2pi, length=40)
    @series begin
        linestyle := :dash
        color := :black
        cos.(ar), sin.(ar)
    end
    @series begin
        seriestype := :scatter
        real.(e), imag.(e)
    end
end

function train(loss, ps, dataset, opt; cb=i->nothing)
    @progress for (i,d) in enumerate(dataset)
        # Flux.reset!(model)
        (l1,l2), back = Zygote.forward(()->loss(d[1]), ps)
        grads = back((1f0,1f0))
        push!(loss1, l1)
        push!(loss2, l2)
        Flux.Optimise.update!(opt, ps, grads)
        cb(i)
    end
end

const h = 0.1

function pendcart(xd,x,p,t)
    g = 9.82; l = 1.0; d = 0.5
    u = 0
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
    xd
end

function generate_data(T=20)
    u0 = @. [pi, 6] * 2 *(rand()-0.5)
    tspan = (0.0,T)
    prob = ODEProblem(pendcart,u0,tspan)
    sol = solve(prob,Tsit5())
    z = reduce(hcat, sol(0:h:T).u)
    y = vcat(sin.(z[1:1,:]), cos.(z[1:1,:])) .+ 0.05 .* randn.()
    z,y
end


##
# function dostuff()
trajs = [generate_data(10) for i = 1:50]
const dataset = [(t[2], t[1]) for t in trajs]

const nz = 3
const nx = size(trajs[1][2], 1)
const nu = 0
const nα = 5

const pxz = Chain(Dense(nz,30,relu), Dense(30,nx))
const f = Chain(Dense(nx+nz+nu,30,relu), Dense(30,2*nz))
const αnet = Chain(Dense(nz,nα), softmax)
# q = Chain(Dense())
const initialw = Chain(Dense(3nx,30,relu), Dense(30,2*nz))
const initialz = Chain(Dense(nz,30,relu), Dense(30,nz))

const A = [(0.001randn(nz,nz)) for _ = 1:nα]
const C = [(0.001randn(nz,nz)) for _ = 1:nα]
const lσp = zeros(nz) # log space
const μp  = zeros(nz)
const lσx = (zeros(nx)) # log space

pars = params((
pxz,
f,
αnet,
initialw,
initialz,
(A...),
(C...),
lσx,
))

##
const c = [0.2]
Zygote.@adjoint function Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector})
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)))
end
Zygote.@adjoint function view(x::AbstractArray, inds...; kwargs...)
    view(x, inds...; kwargs...), dy -> begin
        dx = Zygote._zero(x)
        if dy isa AbstractVector
            copyto!(view(dx, inds...; kwargs...), dy)
        else
            copyto!(view(dx, inds...; kwargs...), view(dy.parent, inds...; kwargs...))
        end
        (dx, ntuple(_->nothing, length(inds))...)
    end
end
@views function finitialw(x)
    μσw = initialw(x[:,1:3][:])
    μw  = μσw[1:nz]
    σw = μσw[(nz+1):2nz]
    varw = σw .^ 2 .+ 1e-5
    w   = μw .+ σw .* randn.()

    return w, μw, varw
end

function lossindividual(x)
    w, μw, σw² = finitialw(x)
    z  = Float64.(initialz(w))
    l0 = kl_normalgauss(μw, σw², c[])
    σx  = exp.(lσx) .+ 1e-5
    sum(1:size(x,2)-1) do i
        xi  = x[:,i]
        # xi1 = x[:,i+1]
        μx  = pxz(z)
        lrec  = sum(abs2(xi[i] - μx[i])/(2.0 * σx[i]^2) + 0.5 * log(2π) + lσx[i]  for i in eachindex(xi))
        μσq = f([z; x[:,i+1]])
        μq  = μσq[1:nz]
        σq  = μσq[(nz+1):2nz]
        w   = μq .+ σq .* randn(nz)

        α  = αnet(z)
        Ai = sum(α[i]*A[i] for i = 1:nα)
        Ci = sum(α[i]*C[i] for i = 1:nα)
        z  = Ai*z + Ci*w
        lkl = kl_normalgauss(μq, σq .^ 2 .+ 1e-5, c[])
         [c[] * lrec + lkl, lrec, kl_normalgauss(μq, σq .^ 2 .+ 1e-5)]
    end + [l0, 0.0, l0]
end

function loss(y)
    lossindividual(y)[1]
end

function kl_normalgauss(μ, σ², c = 1)
    l2π = log(2π)
    lσ = log.(sqrt.(σ²))
    0.5 * sum(c*(l2π) - (l2π + 2lσ[i]) +
    c*(exp(2lσ[i]) + abs2(μ[i])) - 1.0 for i in eachindex(μ))
end

function kl(μ1, lσ1, μ2, lσ2, c = 1)
    l2π = log(2π)
    0.5 * sum(c*(l2π + 2lσ2[i]) - (l2π + 2lσ1[i]) +
    c*(exp(2lσ1[i]) + abs2(μ1[i] - μ2[i]))/(exp(2lσ2[i]) + 1e-5) - 1.0 for i in eachindex(μ1))
end

simulate(T,x::AbstractMatrix) = simulate(T, finitialw(x)[1])

function simulate(T, w = exp.(lσp) .* randn(nz))
    z,x = [],[]
    z = [(initialz(w))]
    for i = 1:T
        μx = pxz(z[end])
        push!(x, (μx))# .+ exp.(σx) .* randn(nx)))

        μσq = f([z[end]; x[end]]) # Wrong time index of xsample?
        μq = μσq[1:end÷2]
        σq = μσq[(end÷2+1):end]
        wsample = μq# .+ σq .* randn(nz)

        α = αnet(z[end])
        Ai = sum(α[i]*A[i] for i = 1:nα)
        Ci = sum(α[i]*C[i] for i = 1:nα)
        push!(z, (Ai*z[end] + Ci*wsample))
    end
    copy(hcat(z...)'),copy(hcat(x...)')
end

function dvbf(x)
    w = finitialw(x)[1]
    z,xf = [],[]
    z = [(initialz(w))]
    for i = 1:size(x,2)-1
        μx = pxz(z[end])
        push!(xf, (μx))# .+ exp.(σx) .* randn(nx)))
        μσq = f([z[end]; x[:,+1]]) # Wrong time index of xsample?
        μq = μσq[1:end÷2]
        σq = μσq[(end÷2+1):end]
        wsample = μq# .+ σq .* randn(nz)

        α = αnet(z[end])
        Ai = sum(α[i]*A[i] for i = 1:nα)
        Ci = sum(α[i]*C[i] for i = 1:nα)
        push!(z, (Ai*z[end] + Ci*wsample))
    end
    copy(hcat(z[1:end-1]...)'),copy(hcat(xf...)')

end



losses = [lossindividual(dataset[1]...)]
opt = ADADelta(0.1)
# Flux.Optimise.updaterule(opt, pars)
Zygote.refresh()
Zygote.gradient(()->lossindividual(dataset[1]...)[1], pars)
@progress for i = 1:20
    global c
    # initializer()
    train(loss, pars, dataset, opt)
    Ta  = 100
    c[] = min(1, c[] + 1/Ta)

    if (i < 50 && i % 5 == 0) || i % 25 == 0
        l = mean(1:50) do i
            (lossindividual(dataset[i]...))
        end
        push!(losses, l)
        @printf("Iter : %d  Loss: %8.4g Rec: %8.4g KL: %8.4g\n", i, (l[1]), (l[2]), (l[3]));
        #plot(losses, layout=1, subplot=1, xscale=:log10, size=(600,400)) |> display
        #     # plot!(dataset[1][2]', subplot=2, c=:blue)
        #     # plot!(sim(dataset[1][1])', subplot=2, c=:orange)
        #     # plot!(data(model(dataset[1][1])'), subplot=2, c=:green, ylims=extrema(dataset[1][2]))
        #     #
        #     # # plot!(data(model(dataset[1][1])'), subplot=1, lab="yhat")
        #     # e = data(rnn.cell.Wh) |> eigvals# .|> log
        #     # scatter!(real.(eA), imag.(eA), lab="Eigenvalues A", subplot=3)
        #     # scatter!(real.(e), imag.(e), lab="Eigenvalues Wh", subplot=3)
        #     # alpha = range(0, stop=2pi, length=30)
        #     # plot!(cos.(alpha), sin.(alpha), l=:dash, subplot=3)#|> display
        #     # gui()
        fo = open("/local/work/fredrikb/res_$(splitdir(@__FILE__)[2])_$(now())", "w");
        serialize(fo, (trajs, dataset, pxz, f, αnet, initialw, initialz, A, C, lσx, opt));
        close(fo);
    end
end
println("Done")

fo = open("/local/work/fredrikb/res_DVBF.jl_2018-10-21T14:21:40.489")
( trajs, dataset, pxz, f, αnet, initialw, initialz, A, C, lσx, opt) = deserialize(fo)
##
z,x = simulate(300)
plot3d(mapslices(x->[x],z,dims=1)..., c=:orange, grid=true)
# plot(losses)
##
i = rand(1:length(dataset))
z,x = simulate(100, dataset[i][1])
plot(z, l=(:orange, ), lab="Filter state")
plot!(x, c=:red, lab="Filter meas")
plot!(dataset[i][2]', c=:blue, lab="State")
plot!(dataset[i][1]', c=:green, lab="Meas")

##

i = rand(1:length(dataset))
θ = dataset[i][2][1,:]
θv = dataset[i][2][2,:]
zf,xf = dvbf(dataset[i][1])

plot(trajs[i][1]', l=(:blue,), layout=2, subplot=1)
plot!(zf, l=(:dash,:orange), subplot=1)
plot!(xf, l=(:red, :dash), subplot=1)
scatter3d!(zf[:,1],zf[:,2],zf[:,3], subplot=2, zcolor=θ)

display(current())


##
plotangle = plot()
plotvelocity = plot()
for i in eachindex(dataset)
    θ = dataset[i][2][1,2:end]
    θv = dataset[i][2][2,2:end]
    zf,xf = dvbf(dataset[i][1])
    scatter3d!(plotangle, zf[2:end,1],zf[2:end,2],zf[2:end,3], zcolor=θ)
    scatter3d!(plotvelocity, zf[2:end,1],zf[2:end,2],zf[2:end,3], zcolor=θv)
end
display(current())
savefig(plotangle, "angle.pdf")
savefig(plotvelocity, "velocity.pdf")

##

θa = Float64[]; θva = Float64[]; zfa = []
for i in eachindex(dataset)
    global θa, θva, zfa
    θ = dataset[i][2][1,:]
    θv = dataset[i][2][2,:]
    zf,xf = dvbf(dataset[i][1])
    θa = vcat(θa, θ)
    θva = vcat(θva, θv)
    push!(zfa, zf)
end
zfa = reduce(vcat, zfa)

k = zfa\θa
kv = zfa\θva
scatter(θa, zfa)
scatter(zfa*k, θa, layout=2, xlabel="\$\\theta\$")
scatter!(zfa*kv, θva, subplot=2, xlabel="\$\\dot\\theta\$")



##
plot()
foreach(eachindex(A)) do i
    eigvalplot!(A[i])
end
display(current())



# avec = range(0, stop=2pi, length=50)
# vvec = range(-3, stop=3, length=50)
# Avec = repeat(avec, 1, 50)
# Vvec = repeat(vvec', 50, 1)
