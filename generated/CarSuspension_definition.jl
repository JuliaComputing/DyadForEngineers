### DO NOT EDIT THIS FILE
### This file is auto-generated by the Dyad command-line compiler.
### If you edit this code it is likely to get overwritten.
### Instead, update the Dyad source code and regenerate this file


@doc Markdown.doc"""
   CarSuspension(; name, wheel_mass, wheel_stiffness, wheel_damping, car_mass, suspension_stiffness, suspension_damping, human_and_seat_mass, seat_stiffness, seat_damping, wheel_initial_position, suspension_initial_position, seat_initial_position)

## Parameters: 

| Name         | Description                         | Units  |   Default value |
| ------------ | ----------------------------------- | ------ | --------------- |
| `wheel_mass`         |                          | kg  |   25 |
| `wheel_stiffness`         |                          | N/m  |   100 |
| `wheel_damping`         |                          | s-1  |   10000 |
| `car_mass`         |                          | kg  |   1000 |
| `suspension_stiffness`         |                          | N/m  |   10000 |
| `suspension_damping`         |                          | s-1  |   10 |
| `human_and_seat_mass`         |                          | kg  |   100 |
| `seat_stiffness`         |                          | N/m  |   1000 |
| `seat_damping`         |                          | s-1  |   50 |
| `wheel_initial_position`         |                          | m  |   0.5 |
| `suspension_initial_position`         |                          | m  |   1 |
| `seat_initial_position`         |                          | m  |   1.5 |
"""
@component function CarSuspension(; name, wheel_mass=25, wheel_stiffness=100, wheel_damping=10000, car_mass=1000, suspension_stiffness=10000, suspension_damping=10, human_and_seat_mass=100, seat_stiffness=1000, seat_damping=50, wheel_initial_position=0.5, suspension_initial_position=1, seat_initial_position=1.5)

  ### Symbolic Parameters
  __params = Any[]
  append!(__params, @parameters (wheel_mass::Float64 = wheel_mass))
  append!(__params, @parameters (wheel_stiffness::Float64 = wheel_stiffness))
  append!(__params, @parameters (wheel_damping::Float64 = wheel_damping))
  append!(__params, @parameters (car_mass::Float64 = car_mass))
  append!(__params, @parameters (suspension_stiffness::Float64 = suspension_stiffness))
  append!(__params, @parameters (suspension_damping::Float64 = suspension_damping))
  append!(__params, @parameters (human_and_seat_mass::Float64 = human_and_seat_mass))
  append!(__params, @parameters (seat_stiffness::Float64 = seat_stiffness))
  append!(__params, @parameters (seat_damping::Float64 = seat_damping))
  append!(__params, @parameters (wheel_initial_position::Float64 = wheel_initial_position))
  append!(__params, @parameters (suspension_initial_position::Float64 = suspension_initial_position))
  append!(__params, @parameters (seat_initial_position::Float64 = seat_initial_position))

  ### Variables
  __vars = Any[]

  ### Constants
  __constants = Any[]

  ### Components
  __systems = ODESystem[]
  push!(__systems, @named wheel = DyadForEngineers.MassSpringDamper(m=wheel_mass, d=wheel_damping, c=wheel_stiffness, g=-10, theta=pi / 2, s0=wheel_initial_position))
  push!(__systems, @named car_and_suspension = DyadForEngineers.MassSpringDamper(m=car_mass, d=suspension_damping, c=suspension_stiffness, g=-10, theta=pi / 2, s0=suspension_initial_position))
  push!(__systems, @named seat = DyadForEngineers.MassSpringDamper(m=human_and_seat_mass, d=seat_damping, c=seat_stiffness, g=-10, theta=pi / 2, s0=seat_initial_position))
  push!(__systems, @named road_data = DyadForEngineers.RoadData())
  push!(__systems, @named road = DyadForEngineers.SimplePosition())

  ### Defaults
  __defaults = Dict()

  ### Initialization Equations
  __initialization_eqs = []

  ### Equations
  __eqs = Equation[]
  push!(__eqs, road.s ~ road_data.y)
  push!(__eqs, connect(road.flange, wheel.flange_sd))
  push!(__eqs, connect(wheel.flange_m, car_and_suspension.flange_sd))
  push!(__eqs, connect(car_and_suspension.flange_m, seat.flange_sd))

  # Return completely constructed ODESystem
  return ODESystem(__eqs, t, __vars, __params; systems=__systems, defaults=__defaults, name, initialization_eqs=__initialization_eqs)
end
export CarSuspension

Base.show(io::IO, a::MIME"image/svg+xml", t::typeof(CarSuspension)) = print(io,
  """<div style="height: 100%; width: 100%; background-color: white"><div style="margin: auto; height: 500px; width: 500px; padding: 200px"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1000 1000"
    overflow="visible" shape-rendering="geometricPrecision" text-rendering="geometricPrecision">
      <defs>
        <filter id='red-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#ff0000" flood-opacity="0.5"/></filter>
        <filter id='green-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#00ff00" flood-opacity="0.5"/></filter>
        <filter id='blue-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="100" flood-color="#0000ff" flood-opacity="0.5"/></filter>
        <filter id='drop-shadow' color-interpolation-filters="sRGB"><feDropShadow dx="0" dy="0" stdDeviation="40" flood-opacity="0.5"/></filter>
      </defs>
    
      </svg></div></div>""")
