using DyadForEngineers

using ForwardDiff
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations
using Plots

#######################################################
###### Problem 1.1: Solving MSD with Julia ######

d = 1
k = 1000
Δt = 1e-3
F = 100

x = zeros(10)

function f(xᵢ, xᵢ₋₁)

    ẋᵢ = (xᵢ - xᵢ₋₁)/Δt     # finite difference derivative
    lhs = d*ẋᵢ + k*xᵢ       # lhs --> left hand side
    rhs = F                 # rhs --> right hand side

    return lhs - rhs        # equation --> lhs = rhs, residual --> 0 = lhs - rhs
end

# f(1, x[1])
# f(-1, x[1])
# f(.1, x[1])
# f(-.1, x[1])
# f(.05, x[1])

# Newton's Method + Implicit/Backwards Euler method
tol = 1e-3
for i=2:10
    g(xᵢ) = f(xᵢ, x[i-1])
    Δx = Inf
    while abs(Δx) > tol
        Δx = g(x[i])/ForwardDiff.derivative(g, x[i])
        x[i] -= Δx
    end
end

plot(x, xlabel="time", ylabel="x")

#######################################################
###### Problem 1.2: Solving MSD with MTK ######

typeof(t)
typeof(D)
sin(t)
D(sin(t))

D(sin(t)) |> expand_derivatives

@mtkmodel SimpleMSD_MTK begin
    @parameters begin
        F = 100
        d = 1
        k = 1000
    end
    @variables begin
        x(t) = 0.0
        ẋ(t) = F/d
    end
    @components begin
        
    end
    @equations begin
        D(x) ~ ẋ
        d*ẋ + k*x ~ F
    end
end

@mtkbuild msd_mtk = SimpleMSD_MTK()

# unknowns(odesys)
# observed(odesys)
# equations(odesys)
# full_equations(odesys)

u0 = []     # <-- used to override defaults of ODESystem variables
p = []      # <-- used to override defaults of ODESystem parameters
tspan = (0.0, 0.01) # solution time span

prob1 = ODEProblem(msd_mtk, u0, tspan, p)
sol1 = solve(prob1)

plot(sol1; idxs=[msd_mtk.x], xlabel="time", ylabel="x")
plot(sol1; idxs=[msd_mtk.ẋ], xlabel="time", ylabel="ẋ")

plot(sol1; idxs=[msd_mtk.x*msd_mtk.k], label="spring force", xlabel="time", ylabel="force")     # spring force
plot!(sol1; idxs=[msd_mtk.ẋ*msd_mtk.d], label="damping force")        # damping force  
plot!(sol1; idxs=[msd_mtk.x*msd_mtk.k + msd_mtk.ẋ*msd_mtk.d], label="spring + damping force")  


#######################################################
###### Problem 1.3: Solving MSD with Dyad ######

@mtkbuild msd_dyad = SimpleMSD_Dyad()
prob2 = ODEProblem(msd_dyad, [], (0, .01), [])
sol2 = solve(prob2)

plot(sol2; idxs=[msd_dyad.x], xlabel="time", ylabel="x")
plot(sol2; idxs=[msd_dyad.x_dot], xlabel="time", ylabel="x_dot")


#######################################################
###### Problem 2: Component-Based Modeling with Dyad ######

@mtkbuild CS = CarSuspension()
prob3 = ODEProblem(CS, [], (0, 10), [])
sol3 = solve(prob3)

plot(sol3; idxs=[CS.seat.mass.s, CS.road_data.y])
plot(sol3; idxs=[CS.seat.mass.s, CS.car_and_suspension.mass.s, CS.wheel.mass.s, CS.road_data.y])