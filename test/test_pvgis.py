"Pvgis test package"

import unittest
from BuildingEnergySimulation import pvgis


class Test_pvgis(unittest.TestCase):
    def setUp(self):
        pass
    def test_url_gen(self):
        targeturl = ('https://re.jrc.ec.europa.eu/pvgis5/seriescalc.php?lat'
                     '=52.5170365&lon=+13.3888599&raddatabase=PVGIS-COSMO&'
                     'userhorizon=&usehorizon=0&angle=0&azimuth=0&startyear='
                     '2007&endyear=2008&mountingplace=building&select_database'
                     '_hourly=PVGIS-COSMO&pvtechchoice=crystSi&peakpower=1&los'
                     's=0&components=1')
        test_loc = [52.5170365, 13.3888599]
        self.assertEqual(pvgis.Pvgis.gen_Pvgis_hourly_url(test_loc),
                         targeturl)
    def test_url_gen_2(self):
        targeturl = ('https://re.jrc.ec.europa.eu/pvgis5/seriescalc.php?lat'
                     '=52.5170365&lon=+13.3888599&raddatabase=PVGIS-COSMO&'
                     'userhorizon=&usehorizon=0&angle=0&azimuth=0&startyear='
                     '2007&endyear=2008&mountingplace=building&select_database'
                     '_hourly=PVGIS-COSMO&pvtechchoice=crystSi&peakpower=1&los'
                     's=0&components=1')
        test_loc = [52.5170365, 13.3888599]
        self.assertEqual(pvgis.Pvgis.gen_Pvgis_hourly_url(test_loc, 
                                                          startyear=2007,
                                                          endyear=2008),
                         targeturl)

if __name__ == '__main__':
    unittest.main()