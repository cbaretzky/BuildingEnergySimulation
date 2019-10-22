.. \_quickstart:

Quickstart
==========

.. code:: python

   import BuildingEnergySimulation as bes
   building= bes.Building(loc="Markt, 52062 Aachen")

Will generate the building as a frame for all components. The following
steps are about adding the relevant components to the building:

Walls
^^^^^

Walls are part of the thermal hull of the building. The majority of
thermal energy is usually transfered through walls.

Adding a wall to the Building can be done as follows:

.. code:: python

   bes.Wall.reg(building, bes.DEFAULT_LAYERS['Brick_Granite'], area=400)

the layers for walls and windows are lists of dictionary containing the
relevant physical parameters to simulate the thermal properties in one
dimension: The “Brick_Granite” layer from the integrated sample layers
is defined as

.. code:: python

   [{
   'c_v': 1000.0, #Heat capacity in J/kg
   'lambda': 0.79, #Heat resistance in W/m²K
   'name': 'Brick', #Name
   'rho': 1800.0, #Density in kg/m³
   'thickness': 0.3}, #Thickness of the layer in m
   {
   'c_v': 1000.0,
   'lambda': 2.80,
   'name': 'Granite',
   'rho': 2600.0,
   'thickness': 0.4}],

The area parameter defines the wall area in square meters

Windows
^^^^

Windows are relevant for thermal losses as well as thermal gains through
solar irradiance. To add a window it is necessary to define the
orientation. azimuth = 0 means the window is facing south, 90 means the
window is facing west and -90 means the window is facing east.

.. code:: python

   bes.Window.reg(building, area= 20, azimuth=0)

Heating
^^^^^^^

Adding a heatpump modeled after one specific groundwater based model

.. code:: python

   bes.Heatpump.reg(building)

Solar-pv and battery
^^^^^^^^^^^^^^^^^^^^

Adding a simplified solar-pv panel with 14 kWp and a battery with 14kWh
capacity and a powergrid connection:

.. code:: python

   bes.Solar_pv_simple.reg(building, kwp = 14,azimuth = 0, inclination=35)
   bes.Battery.reg(building, capacity=14*3.6e6)
   bes.Grid.reg(building)

Connecting all components
^^^^^^^^^^^^^^^^^^^^^^^^^

Finally the last components need to be connected with each other. The
solar panel and heatpump run through the battery, the battery is
connected to the grid.

.. code:: python

   battery = building.get_component('Battery')[0]
   battery.connection_in.append(building.get_component('Solar_pv_simple')[0])
   battery.connection_in.append(building.get_component('Heatpump')[0])
   grid = building.get_component('Grid')[0]
   grid.connection_in.append(battery)

Now the simulation form January first to January fifth of 2007 can be
run as follows

.. code:: python

   building.simulate('2007-1-1', '2007-1-5')

When the processing is finished

.. code:: python

   building.sim_results

contains a pandas Dataframe with all relevant quantities throughout the
simulation timeframe and can be written to disk using the functionality
integrated into pandas ## How it works

The idea is to model the energy flow as a directed graph from the given
environmental factors (outer temperature, solar irradiance) to the
available energy sources (grid, fuel).

Physically speaking the simulation is very inaccurate, however it is balanced around how much information about buildings is typically available and the significant influence of inhabitant habits. For example there is no radiator model. But the heat transfer from a underfloor radiator depends on the position and quantity of furniture in the room. It is much easier to assume a given temperature that the room is supposed to have, compared to of solving a differential equation which greatly depends on individual factors which are rarely accessible.

The main goal is to simulate the energy flows on a timescale that can account for the variability of renewable energies.
