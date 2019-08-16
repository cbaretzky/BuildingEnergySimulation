import requests
import tempfile
import gzip
from pathlib import Path
import os
from hashlib import md5


class Component():
    """
    Helper class to manage connections
    component: component name
    
    """
    def __init__(self, building, connector_in= [], connector_out = [], connection_in=None,):
        self.building = building
        
        self.connector_in =connector_in #list needed connectors
        self.connector_out = connector_out #list available output connectors
        if connection_in is None:
            self.connection_in = [] #components it is connected to
        else:
            self.connection_in = connection_in
        self.description = None
        self.name = "{}_at_{}".format(type(self).__name__, hex(id(self)))
        self.date = None
        self.warn = 0
    @property
    def out(self):
        #Check date
        run_arg = {}
        if self.date == self.building.date:
            return self._out
        else:
            for connector in self.connector_in: #loop over all needed inputs
                values = []
                for connection in self.connection_in: #loop over all connections
                    if connector in connection.connector_out: #if the connector is part of the connection.connection_out, use it
                        values.append(connection.out[connector])
                if len(values)>0:
                    run_arg.update({connector: sum(values)})
                else:
                    if self.warn == 0:
                        print("No usable inputs {} found for {}".format(connector, self.name))
                        self.warn =1
            self.date = self.building.date
        self._out = self.run(**run_arg)
        for key, val in self._out.items():
            if not (key == "name"):
                self.building.result.update({key+'_of_'+self.name : val})
        return self._out
    

class empty_req():
    def __init__(self, content):
        self.content=content

        

def req(url, filecache = True):
    """
    Helper function to cache the result of http requests
    
    """
    if not filecache:
        response = requests.get(url)
        return response
    else:
        reqhash = md5(bytes(url, 'utf')).hexdigest()
        temp_dir = Path(tempfile.gettempdir())
        fname = Path(reqhash+'.pvgis_cache')
        if Path.exists(Path.joinpath(temp_dir,fname)):
            print("Using cached version of {}".format(url))
            with gzip.open(Path.joinpath(temp_dir,fname),'rb') as f:
                resp = empty_req(f.read())
            return resp
        else:
            resp = req(url, filecache = False)
            with gzip.open(Path.joinpath(temp_dir,fname),'wb') as f:
                f.write(resp.content)
            return resp