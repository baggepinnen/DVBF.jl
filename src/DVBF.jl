cd(@__DIR__)
using Plots, Flux, LinearAlgebra, Parameters, Statistics, Random, Printf, OrdinaryDiffEq, DSP, SparseArrays, Dates
using Flux: params, jacobian, train!, data
using Flux.Optimise: Param, optimiser, expdecay #src
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

const h = 0.03

function lorenz(du,u,p,t)
    du[1] = 10.0(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

function pendcart(xd,x,p,t)
    g = 9.82; l = 0.35; d = 1
    u = 0
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
    xd
end

function generate_data(T=20)
    u0 = @. [pi, 1] * 2 *(rand()-0.5)
    tspan = (0.0,T)
    prob = ODEProblem(pendcart,u0,tspan)
    sol = solve(prob,Tsit5())
    z = reduce(hcat, sol(0:h:T).u)
    # z[1,:] .= z[1,:]
    y = z[1:1,:] .+ 0.5 .* randn.()
    # y = copy(x)
    z,y
end


##
# function dostuff()
trajs = [generate_data(3) for i = 1:4]
const dataset = [(t[2], t[1]) for t in trajs]

const nz = 3
const nx = size(trajs[1][2], 1)
const nu = 0
const nα = 5

const pxz = Chain(Dense(nz,30,relu), Dense(30,nx))
const f = Chain(Dense(nx+nz+nu,30,relu), Dense(30,2*nz))
const αnet = Chain(Dense(nz,nα), softmax)
# q = Chain(Dense())
const initialw = Chain(Dense(3nx,30,relu), Dense(30,nz))
const initialz = Chain(Dense(nz,30,relu), Dense(30,nz))

const A = [param(0.1randn(nz,nz)) for _ = 1:nα]
const C = [param(0.1randn(nz,nz)) for _ = 1:nα]
const lσp = zeros(nz) # log space
const μp  = zeros(nz)
const lσx = param(zeros(nx)) # log space

pars = Flux.Tracker.Params([
params(pxz);
params(f);
params(αnet);
params(initialw);
params(initialz);
params(A...);
params(C...);
# params(σx)
])

##
const c = [0.2]

function loss(x,y)
    w   = initialw(x[:,1:3][:])
    z   = initialz(w)
    # σx  = exp.(lσx)
    sum(1:size(x,2)-1) do i
        xi  = x[:,i]
        # xi1 = x[:,i+1]
        μx  = pxz(z) # TODO: How to train this?
        # l1  = sum(abs2, xi .- μx)
        l1  = sum(abs2(xi[i] - μx[i]) for i in eachindex(xi)) #./(σx.^2) .+ log(2π).* 2lσx) # Is this correct?

        μσq = f([z; xi]) # TODO: I change from x[t+1] to x[t]
        μq  = μσq[1:end÷2]
        lσq = μσq[(end÷2+1):end]
        w   = μq .+ exp.(lσq) .* randn() # Why not μq == 0?

        α  = αnet(z)
        Ai = sum(α[i]*A[i] for i = 1:nα)
        Ci = sum(α[i]*C[i] for i = 1:nα)
        z  = Ai*z + Ci*w # Use wsample or μq?
        l2 = kl(μq, lσq, μp, lσp, c[])
        l1 + l2
    end
end

function kl(μ1, lσ1, μ2, lσ2, c = 1)
    l2π = log(2π)
    sum(c*(l2π + 2lσ2[i]) - (l2π + 2lσ1[i]) +
    c*(exp(2lσ1[i]) + abs2(μ1[i] - μ2[i]))/exp(2lσ2[i]) for i in eachindex(μ1))
end

# function kl(μ1, lσ1, μ2, lσ2, c = 1)
#     l2π = log(2π)
#     sum(c*(l2π + 2lσ2[i]) - (l2π + 2lσ1[i]) +
#     c*(exp(2lσ1[i]) + abs2(μ1[i] - μ2[i]))/exp(2lσ2[i]) for i in eachindex(μ1))
# end

simulate(T,x::AbstractMatrix) = simulate(T, initialw(x[:,1:3][:]))

function simulate(T, w = exp.(lσp) .* randn(nz))
    z,x = [],[]
    z = [data(initialz(w))]
    for i = 1:T
        μx = pxz(z[end])
        push!(x, data(μx))# .+ exp.(σx) .* randn(nx)))

        μσq = f([z[end]; x[end]]) # Wrongtime index of xsample?
        μq = μσq[1:end÷2]
        lσq = μσq[(end÷2+1):end]
        wsample = μq #.+ noise*exp.(lσq) .* randn(nz)

        α = αnet(z[end])
        Ai = sum(α[i]*A[i] for i = 1:nα)
        Ci = sum(α[i]*C[i] for i = 1:nα)
        push!(z, data(Ai*z[end] + Ci*wsample))
    end
    copy(hcat(z...)'),copy(hcat(x...)')
end

function dvbf(x)
    w = exp.(lσp) .* randn(nz)
    w = initialw(x[:,1:3][:])
    z,xf = [],[]
    z = [data(initialz(w))]
    for i = 1:size(x,2)
        μx = pxz(z[end])
        push!(xf, data(μx))# .+ exp.(σx) .* randn(nx)))
        μσq = f([z[end]; x[:,i]]) # Wrong time index of xsample?
        μq = μσq[1:end÷2]
        lσq = μσq[(end÷2+1):end]
        wsample = μq #.+ noise*exp.(lσq) .* randn(nz)

        α = αnet(z[end])
        Ai = sum(α[i]*A[i] for i = 1:nα)
        Ci = sum(α[i]*C[i] for i = 1:nα)
        push!(z, data(Ai*z[end] + Ci*wsample))
    end
    copy(hcat(z[1:end-1]...)'),copy(hcat(xf...)')

end


losses = Float64[]
opt = NADAM(pars, 2e-3, decay=0.000001)
@progress for i = 1:800
    global c
    # initializer()
    train!(loss, dataset, opt)
    Ta  = 500
    c[] = min(1, c[] + 1/Ta)

    if (i < 50 && i % 5 == 0) || i % 25 == 0
        l = mean(1:2) do i
            data(loss(dataset[i]...))
        end
        push!(losses, l)
        @printf("Iter : %d  Loss: %8.4g\n", i, l)
        plot(losses, layout=1, subplot=1, xscale=:log10, size=(600,400)) |> display
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
        serialize("/local/work/fredrikb/res_$(splitdir(@__FILE__)[2])_$(now())",( trajs, dataset, pxz, f, αnet, initialw, initialz, A, C, opt))
    end
end
println("Done")

# ( trajs, dataset, pxz, f, αnet, initialw, initialz, A, C, opt) = deserialize("/local/work/fredrikb/res_DVBF.jl_2018-10-21T14:21:40.489")
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
plot()
for i in eachindex(dataset)
    θ = dataset[i][2][1,:]
    θv = dataset[i][2][2,:]
    zf,xf = dvbf(dataset[i][1])
    scatter3d!(zf[:,1],zf[:,2],zf[:,3], zcolor=θ)
end
display(current())

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
