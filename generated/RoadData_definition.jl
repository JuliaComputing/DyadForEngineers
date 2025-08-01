### DO NOT EDIT THIS FILE
### This file is auto-generated by the Dyad command-line compiler.
### If you edit this code it is likely to get overwritten.
### Instead, update the Dyad source code and regenerate this file


@doc Markdown.doc"""
   RoadData(; name, bump, freq, offset, loop)

## Parameters: 

| Name         | Description                         | Units  |   Default value |
| ------------ | ----------------------------------- | ------ | --------------- |
| `bump`         |                          | --  |   0.2 |
| `freq`         |                          | --  |   0.5 |
| `offset`         |                          | --  |   0.4 |
| `loop`         |                          | --  |   10 |

## Connectors

 * `y` - This connector represents a real signal as an output from a component ([`RealOutput`](@ref))
"""
@component function RoadData(; name, bump=0.2, freq=0.5, offset=0.4, loop=10)

  ### Symbolic Parameters
  __params = Any[]
  append!(__params, @parameters (bump::Float64 = bump))
  append!(__params, @parameters (freq::Float64 = freq))
  append!(__params, @parameters (offset::Float64 = offset))
  append!(__params, @parameters (loop::Float64 = loop))

  ### Variables
  __vars = Any[]
  append!(__vars, @variables y(t), [output = true])

  ### Constants
  __constants = Any[]

  ### Components
  __systems = ODESystem[]

  ### Defaults
  __defaults = Dict()

  ### Initialization Equations
  __initialization_eqs = []

  ### Equations
  __eqs = Equation[]
  push!(__eqs, y ~ ifelse((t % loop) < offset, 0, ifelse((t % loop) - offset > freq, 0, (bump * (1 - cos(2 * pi * (t - offset) / freq))))))

  # Return completely constructed ODESystem
  return ODESystem(__eqs, t, __vars, __params; systems=__systems, defaults=__defaults, name, initialization_eqs=__initialization_eqs)
end
export RoadData

Base.show(io::IO, a::MIME"image/svg+xml", t::typeof(RoadData)) = print(io,
  """<div style="height: 100%; width: 100%; background-color: white"><div style="margin: auto; height: 500px; width: 500px; padding: 200px"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1000 1000"
    overflow="visible" shape-rendering="geometricPrecision" text-rendering="geometricPrecision">
      <defs>
        <filter id='red-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#ff0000" flood-opacity="0.5"/></filter>
        <filter id='green-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#00ff00" flood-opacity="0.5"/></filter>
        <filter id='blue-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#0000ff" flood-opacity="0.5"/></filter>
        <filter id='drop-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="40" flood-opacity="0.5"/></filter>
      </defs>
    
      </svg></div></div>""")
