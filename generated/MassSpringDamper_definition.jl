### DO NOT EDIT THIS FILE
### This file is auto-generated by the Dyad command-line compiler.
### If you edit this code it is likely to get overwritten.
### Instead, update the Dyad source code and regenerate this file


@doc Markdown.doc"""
   MassSpringDamper(; name, m, c, d, g, s0, v0, a0, theta)

## Parameters: 

| Name         | Description                         | Units  |   Default value |
| ------------ | ----------------------------------- | ------ | --------------- |
| `m`         | Component mass                         | kg  |    |
| `c`         | Stiffness of the spring                         | N/m  |    |
| `d`         | Damping coefficient                         | s-1  |    |
| `g`         | Gravity                         | m/s2  |    |
| `s0`         | Initial position                         | m  |    |
| `v0`         | Initial velocity                         | m/s  |   0 |
| `a0`         | Initial acceleration                         | m/s2  |   0 |
| `theta`         | Angle of motion with +X axis                         | rad  |   0 |

## Connectors

 * `flange_m` - This connector represents a mechanical flange with position and force as the potential and flow variables, respectively. ([`Flange`](@ref))
 * `flange_sd` - This connector represents a mechanical flange with position and force as the potential and flow variables, respectively. ([`Flange`](@ref))
"""
@component function MassSpringDamper(; name, m=nothing, c=nothing, d=nothing, g=nothing, s0=nothing, v0=0, a0=0, theta=0)

  ### Symbolic Parameters
  __params = Any[]
  append!(__params, @parameters (m::Float64 = m), [description = "Component mass"])
  append!(__params, @parameters (c::Float64 = c), [description = "Stiffness of the spring"])
  append!(__params, @parameters (d::Float64 = d), [description = "Damping coefficient"])
  append!(__params, @parameters (g::Float64 = g), [description = "Gravity"])
  append!(__params, @parameters (s0::Float64 = s0), [description = "Initial position"])
  append!(__params, @parameters (v0::Float64 = v0), [description = "Initial velocity"])
  append!(__params, @parameters (a0::Float64 = a0), [description = "Initial acceleration"])
  append!(__params, @parameters (theta::Float64 = theta), [description = "Angle of motion with +X axis"])

  ### Variables
  __vars = Any[]

  ### Constants
  __constants = Any[]

  ### Components
  __systems = ODESystem[]
  push!(__systems, @named flange_m = __Dyad__Flange())
  push!(__systems, @named flange_sd = __Dyad__Flange())
  push!(__systems, @named mass = TranslationalComponents.Mass(m=m, L=0, g=g, theta=theta))
  push!(__systems, @named spring_damper = TranslationalComponents.SpringDamper(d=d, c=c))

  ### Defaults
  __defaults = Dict()
  __defaults[mass.s] = (s0)
  __defaults[mass.v] = (v0)
  __defaults[mass.a] = (a0)

  ### Initialization Equations
  __initialization_eqs = []

  ### Equations
  __eqs = Equation[]
  push!(__eqs, connect(flange_m, mass.flange_b, spring_damper.flange_a))
  push!(__eqs, connect(spring_damper.flange_b, flange_sd))

  # Return completely constructed ODESystem
  return ODESystem(__eqs, t, __vars, __params; systems=__systems, defaults=__defaults, name, initialization_eqs=__initialization_eqs)
end
export MassSpringDamper

Base.show(io::IO, a::MIME"image/svg+xml", t::typeof(MassSpringDamper)) = print(io,
  """<div style="height: 100%; width: 100%; background-color: white"><div style="margin: auto; height: 500px; width: 500px; padding: 200px"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1000 1000"
    overflow="visible" shape-rendering="geometricPrecision" text-rendering="geometricPrecision">
      <defs>
        <filter id='red-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#ff0000" flood-opacity="0.5"/></filter>
        <filter id='green-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#00ff00" flood-opacity="0.5"/></filter>
        <filter id='blue-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#0000ff" flood-opacity="0.5"/></filter>
        <filter id='drop-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="40" flood-opacity="0.5"/></filter>
      </defs>
    
      </svg></div></div>""")
