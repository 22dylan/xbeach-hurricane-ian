import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import imageio
# import netCDF4
import seaborn as sns
import xarray as xr



class xb_plotting_large():
    """docstring for xb_plotting_large"""
    def __init__(self, model_runname, var="H"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_runname = model_runname
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "models", self.model_runname)
        self.var = var
        self.read_buildings()

    def read_buildings(self):
        elev = self.read_data_xarray(var="zb", t=0)
        mask = (elev != 99)
        mask = (elev < 10)
        self.bldgs = np.ma.array(elev, mask=mask)

    def plot_timestep(self, t=1, vmax=None, fname=None, prnt_read=False):
        """ function to plot single timestep
        """
        data_plot, time = self.read_data_xarray(var="H", t=t, prnt_read=prnt_read, rtn_time_array=True)
        xgr, ygr = self.read_grid()
        data_shape = np.shape(data_plot)
        if data_shape[1]>data_shape[0]:
            figsize = (8,1)
            txt_x = 0.02
            txt_y = 0.3
        else:
            figsize = (12,8)
            txt_x = 0.1
            txt_y = 0.83

        # fig, ax = plt.subplots(1,1, figsize=figsize)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1,2.8]})

        # --- new
        mask = (data_plot < -99999)
        masked_array = np.ma.array(data_plot, mask=mask)
        
        # setting up colormap for water
        if self.var == "H":
            cmap = mpl.cm.plasma
            cmap.set_bad('bisque')
            vmax = 1.0 if vmax == None else vmax
            vmin = 0
        else:
            cmap = mpl.cm.cividis
            cmap.set_bad('bisque')
            vmax = 3.0 if vmax == None else vmax
            vmin = 0.0

        cmap_bldg = mpl.cm.Greys_r
        cmap_bldg.set_bad(alpha=0)

        if self.var == "H":
            s = "Wave Height (m)\nTime: {:2.0f}h ({:8.0f}s)" .format(time[t]/3600, time[t])
            cbar_s = "Wave Height (m)"
        elif self.var == "zs":
            s = "Water Elevation (m)\nTime: {:2.0f}h ({:8.0f}s)" .format(time[t]/3600, time[t])
            cbar_s = "Water Elevation (m)"
        elif self.var == "zs0":
            s = "Water Elevation - Tide Alone (m) \nTime {:2.0f}h ({:8.0f}s)" .format(time[t]/3600, time[t])
            cbar_s = "Water Elevation - Tide Alone (m)"

        # -- drawing first plot
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax1, extend="max", label=cbar_s)

        ax0.pcolormesh(xgr, ygr, self.bldgs, cmap=cmap_bldg)
        ax0.set_title(s)

        # -- drawing second, zoomed in plot
        # full model domain
        box_lower_left = (2600, 5000)       # in world units
        dx, dy = 1000, 1000

        # # small model domain
        # box_lower_left = (100, 0)       # in world units
        # dx, dy = 100, 100

        # continuing with zommed in plot
        box_upper_right = (box_lower_left[0]+dx, box_lower_left[1]+dy)

        id_ll = self.wrld_to_grid_index(xgr, ygr, box_lower_left)
        id_ur = self.wrld_to_grid_index(xgr, ygr, box_upper_right)
        
        xgr2 = xgr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        ygr2 = ygr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        masked_array2 = masked_array[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        bldgs2 = self.bldgs[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]

        ax1.pcolormesh(xgr2, ygr2, masked_array2, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.pcolormesh(xgr2, ygr2, bldgs2, cmap=cmap_bldg)
        
        # # --old
        box_l = xgr2[0,-1] - xgr2[0,0]
        box_h = ygr2[-1,0] - ygr2[0,0]

        # # -- adding rectangle showing where zoomed in area is
        rect = patches.Rectangle(box_lower_left, box_l, box_h, linewidth=3, zorder=10, edgecolor='r', facecolor='none')
        ax0.add_patch(rect)
        
        # --- saving file
        if fname != None:
            plt.savefig(fname,
                        transparent=False, 
                        dpi=500,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()


    def make_animation_imageio(self, tstart=None, tstop=None, vmax=2, makefigs=True):
        if tstart == None:
            tstart = 0
        if tstop == None:
            tstop = 30
        
        # reading data
        # data_plot = self.data[tstart:tstop,:,:]

        # --- making images to comprise video
        temp_dir = os.path.join(self.file_dir, "temp")
        if makefigs:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            self.make_directory(temp_dir)

            for t in range(tstart, tstop):
                if t%10==0:
                    print(t)
                fn = os.path.join(temp_dir, "f{}.png" .format(t))
                self.plot_timestep(t, fname=fn, vmax=vmax)
                plt.close()
        
        # --- making video
        video_name = '{}-{}.mp4' .format(self.model_runname, self.var)
        writer = imageio.get_writer(video_name, fps=10, format='FFMPEG')
        for step in range(tstart, tstop):
            fn = os.path.join(temp_dir, "f{}.png" .format(step))
            if os.path.isfile(fn):
                image = imageio.v2.imread(fn)
                writer.append_data(image)
        writer.close()

    def wrld_to_grid_index(self, xgr, ygr, xy):
        idx = np.argmin(np.abs(xgr[0,:] - xy[0]))
        idy = np.argmin(np.abs(ygr[:,0] - xy[1]))        
        return (idx,idy)


    def read_data_xarray(self, var, t, prnt_read=False, rtn_time_array=False):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        slice_data = ds[var].isel(globaltime=slice(t,t+1))
        if rtn_time_array:
            time = ds["globaltime"].values
            return slice_data.values[0,:,:], time
        else:
            return slice_data.values[0,:,:]

    def read_grid(self):
        fn = os.path.join(self.path_to_model, "x.grd")
        if os.path.isfile(fn):
            xgrid = os.path.join(self.path_to_model, "x.grd")
            ygrid = os.path.join(self.path_to_model, "y.grd")

            with open(xgrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    xs = [float(i.strip()) for i in line.split()]
                    if cnt == 0:
                        break
                    
            ys = []
            with open(ygrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    y_ = [float(i.strip()) for i in line.split()][0]
                    ys.append(y_)


            xgr, ygr = np.meshgrid(xs, ys)
        else:
            fn_params = os.path.join(self.path_to_model, "params.txt")
            with open(fn_params) as f:
                for cnt, line in enumerate(f.readlines()):
                    ls = [i.strip() for i in line.split()]
                    if "dx" in ls:
                        dx = float(ls[-1])
                    elif "dy" in ls:
                        dy = float(ls[-1])
                    elif "nx" in ls:
                        nx = float(ls[-1])
                    elif "ny" in ls:
                        ny = float(ls[-1])
            
            xs = np.arange(start=0, stop=nx*dx+dx, step=dx)
            ys = np.arange(start=0, stop=ny*dy+dx, step=dy)
            xgr, ygr = np.meshgrid(xs, ys)

        return xgr, ygr

    def make_directory(self, path_out):
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        return path_out

if __name__ == "__main__":
    xbpl = xb_plotting_large(model_runname="test-waves3", var="H")
    # xbpl.read_data_xarray(var="H", t=0, prnt_read=True)
    # xbpl.make_animation_imageio(tstart=0, tstop=75, makefigs=True)
    xbpl.plot_timestep(t=100, vmax=1, prnt_read=True)
    plt.show()








