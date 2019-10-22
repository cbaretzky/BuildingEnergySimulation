"""Main package test"""

import unittest
import BuildingEnergySimulation as bes


class Test_main(unittest.TestCase):
    def setUp(self):
        pass

    def test_small(self):
        building = bes.Building(loc="Markt, 52062 Aachen")
        bes.Wall.reg(building, area=400,
                     layers=bes.DEFAULT_LAYERS['Brick_Granite'],)
        bes.Window.reg(building, area=20, azimuth=0)
        bes.Heatpump.reg(building)
        bes.Solar_pv_simple.reg(building, kwp=14, azimuth=0, inclination=35)
        bes.Battery.reg(building, capacity=14*3.6e6)
        bes.Grid.reg(building)
        battery = building.get_component('Battery')[0]
        battery.connection_in.append(building.get_component(
            'Solar_pv_simple')[0])
        battery.connection_in.append(building.get_component(
            'Heatpump')[0])
        grid = building.get_component('Grid')[0]
        grid.connection_in.append(battery)
        res = building.simulate('2007-1-1', '2007-1-5')
        self.assertEqual(res.shape,  (577, 16))
        self.assertEqual(round(res['Tamb'].mean()), 6)

    def test_big(self):
        building = bes.Building(loc="Markt, 52062 Aachen")
        window_area_south = 4*1.2*2.4
        window_area_west = 2*1.2*2.4
        window_area_north = 3*1.2*2.4
        window_area_east = 3*1.2*2.4
        door_area_west = 1.2*2.4
        door_area_north = 1.2*2.4
        area_wall_south = 8.8*5-window_area_south
        area_wall_west = 4*(5+3.3*.5)-window_area_west-door_area_west
        area_wall_north = 8.8*5-window_area_north-door_area_north
        area_wall_east = 4*(5+3.3*.5)-window_area_east
        area_wall = (area_wall_south+area_wall_west +
                     area_wall_north+area_wall_east)
        area_roof = 2*4.4*8.8
        area_ground_plane = 4*8.8
        bes.Wall.reg(building, area=area_wall,
                     layers=bes.DEFAULT_LAYERS['Brick'], )
        bes.Wall.reg(building, area=area_roof,
                     layers=bes.DEFAULT_LAYERS['Roof'], )
        bes.Ground_floor.reg(building, area=area_ground_plane,
                             layers=bes.DEFAULT_LAYERS['Brick'])
        bes.Wall.reg(building, area=door_area_west+door_area_north,
                     layers=bes.DEFAULT_LAYERS['Roof'], )
        bes.Window.reg(building, area=window_area_south,
                       layers=bes.DEFAULT_LAYERS['Window'], azimuth=0)
        bes.Window.reg(building, area=window_area_west,
                       layers=bes.DEFAULT_LAYERS['Window'], azimuth=90)
        bes.Window.reg(building, area=window_area_north,
                       layers=bes.DEFAULT_LAYERS['Window'], azimuth=180)
        bes.Window.reg(building, area=window_area_east,
                       layers=bes.DEFAULT_LAYERS['Window'], azimuth=-90)
        bes.Heatpump.reg(building)
        bes.Solar_pv_simple.reg(building, kwp=14, azimuth=0, inclination=35)
        bes.Battery.reg(building, capacity=14*3.6e6)
        bes.Grid.reg(building)
        battery = building.get_component('Battery')[0]
        battery.connection_in.append(building.get_component(
            'Solar_pv_simple')[0])
        battery.connection_in.append(building.get_component(
            'Heatpump')[0])
        grid = building.get_component('Grid')[0]
        grid.connection_in.append(battery)
        res = building.simulate('2007-1-1', '2007-1-5')
        self.assertEqual(res.shape,  (577, 25))

        self.assertEqual(round(res['Tamb'].mean()), 6)


if __name__ == '__main__':
    unittest.main()
