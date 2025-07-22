using ModelingToolkit
using OrdinaryDiffEq
using Plots


# ------------------------------------------------
# Part 1: Steady State Modeling ------------------
# ------------------------------------------------

pars = @parameters begin
    ẋ=1 
    A=0.1 
    c=1000 
    pₛ=300e5 
    pᵣ=0 
    ρ=1000 
    Cₒ=2.7 
    m=100
end
vars = @variables begin
    Aₒ=0.001
    u=100
    p₁ 
    p₂ 
end
eqs = [
    u ~ ẋ * (A/Aₒ)
    pₛ - p₁ ~ (1/2)*ρ*u^2*Cₒ
    p₂ - pᵣ ~ (1/2)*ρ*u^2*Cₒ
    0 ~ (p₁ - p₂)*A - c*ẋ
]

@mtkbuild sys = NonlinearSystem(eqs, vars, pars)
prob = NonlinearProblem(sys, [], [])
sol = solve(prob, abstol=1e-9)

sol[Aₒ]
sol[u]
sol[ẋ * (A/Aₒ)]
sol.ps[pₛ]

orifices = []
velocity_limits = 1.0:0.1:2.0
for velocity_limit in velocity_limits
    prob′ = remake(prob; p=[ẋ => velocity_limit])
    sol′ = solve(prob′; abstol=1e-9)
    push!(orifices, sol′[Aₒ])
end

plot(velocity_limits, orifices; xlabel="velocity limit [m/s]", ylabel="orifice size [m^2]", size=(600,600))


# ------------------------------------------------
# Part 2: Dynamic Modeling (DAEs) ----------------
# ------------------------------------------------

using ModelingToolkit: t_nounits as t, D_nounits as D

# typeof(t)
# typeof(D)
# sin(t)
# D(sin(t))

# D(sin(t)) |> expand_derivatives

pars = @parameters begin
    Aₒ=0.00095 
    A=0.1 
    c=1000 
    pₛ=300e5 
    pᵣ=0 
    ρ=1000 
    Cₒ=2.7 
    m=100
end
vars = @variables begin 
    x(t)=0 
    ẋ(t)=0 
    ẍ(t), [guess=0]
    u(t), [guess=100]
    p₁(t), [guess=0]
    p₂(t), [guess=0]
end
eqs = [
    D(x) ~ ẋ
    D(ẋ) ~ ẍ

    u ~ ẋ * (A/Aₒ)
    pₛ - p₁ ~ (1/2)*ρ*u^2*Cₒ
    p₂ - pᵣ ~ (1/2)*ρ*u^2*Cₒ
    m*ẍ ~ (p₁ - p₂)*A - c*ẋ
]

@mtkbuild sys = ODESystem(eqs, t, vars, pars)
prob = ODEProblem(sys, [], (0.0, 0.0001), [])
sol = solve(prob)


plot(sol, idxs=[x]; ylabel="position [m]", size=(600,600))

plot(sol, idxs=[ẋ]; ylabel="velocity [m/s]", size=(600,600))

plot(sol, idxs=[ẍ]; ylabel="acceleration [m/s^2]", size=(600,600))

plot(sol, idxs=[p₁, p₂]; ylabel="pressure [Pa]", size=(600,600))

plot(sol, idxs=[c*ẋ]; ylabel="damper force [N]", size=(600,600))
