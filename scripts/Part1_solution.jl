using DyadForEngineers

using ForwardDiff
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations
using Plots


#######################################################
###### Problem 1.1: Solving MSD with Julia ######

Δt = 1e-3   # step size
d = 1       # damping coefficient
k = 1000    # spring coefficient
F = 100     # external force

x = zeros(10)

# xᵢ: guess for the position at next timestep
# xᵢ₋₁: position at current timestep
# f returns 0 iff xᵢ and xᵢ₋₁ are part of the same solution
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

# typeof(t)
# sin(t)
# typeof(sin(t))

# typeof(D)
# D(sin(t))
# D(sin(t)) |> expand_derivatives

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
    @equations begin
        d*ẋ + k*x ~ F
        D(x) ~ ẋ
    end
end

@mtkbuild msd_mtk = SimpleMSD_MTK()

# unknowns(msd_mtk)
# observed(msd_mtk)
# equations(msd_mtk)
# full_equations(msd_mtk)

prob1 = ODEProblem(msd_mtk, [], (0.0, 0.01), [])
sol1 = solve(prob1)

plot(sol1; idxs=[msd_mtk.x], xlabel="time", ylabel="x")
plot(sol1; idxs=[msd_mtk.ẋ], xlabel="time", ylabel="ẋ")

plot(sol1; idxs=[msd_mtk.x * msd_mtk.k], label="spring force", xlabel="time", ylabel="force")     # spring force
plot!(sol1; idxs=[msd_mtk.ẋ * msd_mtk.d], label="damping force")        # damping force  
plot!(sol1; idxs=[(msd_mtk.x * msd_mtk.k) + (msd_mtk.ẋ * msd_mtk.d)], label="spring + damping force")  


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