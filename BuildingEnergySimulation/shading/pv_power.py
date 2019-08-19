import io
import requests
import numpy as np
import datetime as dt
import pandas as pd

        
        

def gen_string_configs(module):
    grid_start_x = 10
    grid_start_y = 2
    grid_len_x = 150/10
    grid_len_y = 74/10
    module_x = module['dimx']
    module_y = module['dimy']
    grid_shape = (int(grid_len_y//module_y),int(grid_len_x//module_x))
    
    grid= -np.ones(grid_shape)
    grid[:,:2] = 0
    #grid[:,6] = 2
    #grid[:,7] = 2
    grid[:,-2:] = 1
    grid[-1,10:12] = 1
    grid[-1,0:10] = 2
    print(grid.shape)
    string_configs = {'1': {'population' : np.argwhere(grid==0), 
                           'grid' : grid==0},
                     '2': {'population' : np.argwhere(grid==1), 
                           'grid' : grid==1},
                     '3': {'population' : np.argwhere(grid==2), 
                           'grid' : grid==2}
                     }
    return (grid_start_x, grid_start_y, module, string_configs,grid )
  
module = {
    'dimx' : 1.046,
    'dimy' : 1.558,
    'cells_x' : 6,
    'cells_y' : 10,
    'cell_size' : 15.24,
    'power' : 370,
    'cell_area'  : 153.33,
    'vbypass' : -.5,
    'bypass_config' : [12, [2,4,2]],
    'i_sc':6.48,
    'v_oc':69.5/96,
    'i_mp':6.09,
    'v_mp' : 59.1/96,
    }


def get_roof(date):
    """
    Return roof image for sun position on date
    az = azimuth (0 at southern direction)
    """
    key = date.strftime('%m_%d_%H_')+"{:02d}".format(int(date.minute//10*10))
    az, al = suns[key].az_al
    #print(suns[key].az_al)
    az=int(az)
    al=int(al)
    if not suns[key].up:
        return False
    try:
        return roof_dict[str(az)+"_"+str(al)+'.png']
    except:
        print('couldn_t find '+str(az)+"_"+str(al)+'.png at time '+str(date))
        return False

class solar_array():
    def __init__(self, x_min,y_min , solar_module,  string_configs,grid, I_conf =np.linspace(-4,7,100) ):
        self.x_min = x_min
        self.y_min = y_min
        self.ind = 0 #index of max cell in subcells
        self.grid= grid
        self.module = solar_module
        self.I_conf = I_conf
        self.strings =[]
        self.string_configs= string_configs
        self.module_pos = []
        self.moduledict = {}
        self.L_mul_L = []
        self.dates = []
        self.timestep = 600
        self.batches = []
        for name, string_config in self.string_configs.items():
            #module_array=  string_config['grid'].repeat(module['cells_y'],axis = 0).repeat(module['cells_x'],axis = 1)
            ##GET lists of slices according to bypass for each module 
            #print(string_config)
            for module_coord in string_config['population']:
                L_slice = []
                stripe_length, bypass_grouping = module['bypass_config']
                #dx = np.sum(bypass_grouping[0])
                dy=np.sum(bypass_grouping)
                dx = stripe_length
                #print(dx)
                x= module_coord[0]*dx+self.x_min
                y= module_coord[1]*dy+self.y_min
                #x= module_coord[0]*8
                #y= module_coord[1]*12
                for bypass in bypass_grouping:
                    x_min = x
                    y_min = y 
                    x_max = dx+x
                    y_max = bypass+y
                    L_slice.append([slice(x_min, x_max), slice(y_min, y_max)])
                    y=y_max                    
                self.module_pos.append(L_slice)
    def get_VI_(self, PVGIS, start_date=dt.datetime(2007,1,12), end_date=dt.datetime(2016,1,1),  timestep=600, ):
        
        I_conf = self.I_conf
        steps = int(end_date.timestamp()-start_date.timestamp())//timestep
        #print(end_date.timestamp()-start_date.timestamp())
        n_modules = len(self.module_pos)
        resolution_I = np.size(I_conf)
        self.V_I_data = np.zeros((steps, n_modules,resolution_I )) 
        for time in range(int(end_date.timestamp()-start_date.timestamp())//timestep):
            if time>0 and (time%(steps//10)==0):
                #pass
                print(int(time/steps*100))
            curr_time = start_date+dt.timedelta(seconds=time*timestep)
            environment = PVGIS[curr_time]
            L_mult = environment['Bi']
            L_add_di = environment['Di']
            self.L_mul_L.append([L_mult,L_add_di])
            T_am =environment['Tamb']
            W_10 = environment['W10']
            roof = get_roof(curr_time)
            if not(np.any(roof)):
                continue
            if L_mult == 0:
                continue
            for ind, module in enumerate(self.module_pos):
                L_ = [np.reshape(roof[m_[0],m_[1]],-1)*L_mult*1e-3+L_add_di*1e-3 for m_ in module]
                #go = (I_conf,[np.reshape(roof[m_[0],m_[1]],-1)*L_mult*1e-3 for m_ in module], T_am, W_10, -.5)
                #self.V_I_data[time, ind] = V_module_(go)
                #self.V_I_data[time, ind] = V_module(I_conf,L_,T_am, W_10, rb=-.5)
                self.V_I_data[time, ind] = V_module(I_conf,L_,0, 0, rb=-.5)
    def get_VI_P(self, PVGIS, start_date=dt.datetime(2007,1,1), end_date=dt.datetime(2008,12,31),):
        timestep = self.timestep
        I_conf = self.I_conf
        n_threads = 16
        batch_size = 24*6*7
        steps = int((end_date.timestamp()-start_date.timestamp())/timestep/batch_size)
        #print(end_date.timestamp()-start_date.timestamp())
        n_modules = len(self.module_pos)
        resolution_I = np.size(I_conf)
        self.V_I_data = np.zeros((steps, n_modules,resolution_I )) 
        self.batch_data = []
        self.batches = []
        for time in range(int((end_date.timestamp()-start_date.timestamp())/timestep/batch_size)):
            curr_time = start_date+dt.timedelta(seconds=time*timestep*batch_size)
            end_time = start_date+dt.timedelta(seconds=(time+1)*timestep*batch_size)
            self.batches.append((curr_time, end_time))
        #print(batches)
        
        with Pool(n_threads) as p:
            self.batch_data = p.map(self.process, self.batches)
    def process(self, args):
        timestep = self.timestep
        (start_date, end_date) = args
        I_conf = self.I_conf
        steps = int(end_date.timestamp()-start_date.timestamp())//timestep
        #print(steps)
        resolution_I = np.size(self.I_conf)
        n_modules = len(self.module_pos)
        V_I_data =np.zeros((steps, n_modules,resolution_I )) 
        #print(str(start_date))
        for time in range(int(end_date.timestamp()-start_date.timestamp())//timestep):
            if time>0 and (time%(steps//10)==0):
                pass
                #print(int(time/steps*100))
            curr_time = start_date+dt.timedelta(seconds=time*timestep)
            environment = PVGIS[curr_time]
            L_mult = environment['Bi']
            L_add_di = environment['Di']
            self.L_mul_L.append([L_mult,L_add_di])
            T_am =environment['Tamb']
            W_10 = environment['W10']
            roof = get_roof(curr_time)
            if not(np.any(roof)):
                continue
            if L_mult == 0:
                continue
            for ind, module in enumerate(self.module_pos):
                L_ = [np.reshape(roof[m_[0],m_[1]],-1)*L_mult*1e-3+L_add_di*1e-3 for m_ in module]
                #go = (I_conf,[np.reshape(roof[m_[0],m_[1]],-1)*L_mult*1e-3 for m_ in module], T_am, W_10, -.5)
                #self.V_I_data[time, ind] = V_module_(go)
                #self.V_I_data[time, ind] = V_module(I_conf,L_,T_am, W_10, rb=-.5)
                V_I_data[time, ind] = V_module(I_conf,L_,0, 0, rb=-.5)
            np.save("/home/clemens/data/VI/"+start_date.strftime('%Y-%m-%d'), V_I_data)
        return [start_date, end_date,  V_I_data]


        

            
    
    def get_power_curve_SE(self):
        power_arr = np.zeros(self.V_I_data.shape[0])
        power_arr = self.V_I_data*self.I_conf[None,None,:]
        
        #return np.amax(np.sum(power_arr, axis=1), axis=1)
        return np.sum(np.amax(power_arr, axis=2), axis=1)
        
    def get_power_curve(self):
        power_arr = np.zeros(self.V_I_data.shape[0])
        power_arr = self.V_I_data*self.I_conf[None,None,:]
        
        return n_modulesnp.amin(np.amax(power_arr, axis=1), axis=1)
        
                
    @property
    def bbox(self):
        return np.asarray([self.x_min,self.y_min,self.dimx,self.dimy])
    
    def bbox2(self):
        return np.asarray([self.x_min,self.y_min]),self.dimx,self.dimy
#test1 = solar_array(*gen_string_configs(module))

#dat = pd.read_csv(io.BytesIO(requests.get(gen_PVGIS_hourly(aspect=44, angle=35),).content), #skiprows=10, skipfooter=12, parse_dates=[0], date_parser=PVGIS_date)
#PVGIS = gen_PVGIS(dat)


