import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

class plot_transect():
    """docstring for xb_plotting_pt"""
    def __init__(self, model_runname, var="H"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_runname = model_runname
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "xbeach", "models", self.model_runname)
        self.var = var
        self.xgr, self.ygr = self.read_grid()


    def var2label(self, var):
        if var == "H":
            ylabel = "Wave Height (m)"
        elif var == "zs":
            ylabel = "Water Level (m)"
        elif var == "zs0":
            ylabel = "zs0: Water Level - Surge/Tide Alone (m)"
        elif var=="zs1":
            ylabel = "zs1: Water Level - No Surge/Tide"
        return ylabel
    
    def read_data_xarray_transect(self, var, idy, t, prnt_read=False, rtn_time_array=False):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        time = ds["globaltime"].values
        t_sec = t*3600
        t_idx = np.argmin(np.abs(time - t_sec))

        # slice_data = ds[var].isel(globaltime=slice(t,t+1))
        slice_data = ds[var][t_idx,idy,:]

        return slice_data.values

    def read_data_xarray_gd(self, var="zb"):
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        slice_data = ds[var][0,:,:]
        return slice_data

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


    def plot_water_level_transect(self, y_trans, ts, plot_ground=True, h_plus_zs=False, fulldomain=True, drawdomain=False, fname=None):
        idy = np.argmin(np.abs(self.ygr[:,0] - y_trans))
    
        fig, ax = plt.subplots(1,1,figsize=(10,4))
        colors = sns.color_palette("viridis")
        if plot_ground == True:
            grnd = self.read_data_xarray_transect(var="zb", idy=idy, t=0)

        # get data for variable
        for t_i, t in enumerate(ts):
            data_ = self.read_data_xarray_transect(var=self.var, idy=idy, t=t)
            data_[data_<-99999] = 0
            c = colors[t_i]

            # -- temporary            
            if h_plus_zs:
                data_zs = self.read_data_xarray_transect(var="zs", idy=idy, t=t)
                data_zs[data_<-99999] = 0
                data_tot = data_zs + data_

                ax.plot(data_tot, color=c, lw=2, label="H+zs" .format(t))
                ax.plot(data_zs, color="grey", ls="-.", lw=1, label="zs" .format(t))
                ax.plot(data_, color="green", lw=1, label="H" .format(t))

                s_title = "water elevation at y={} m; t={} hr" .format(y_trans, t)
            else:
                ax.plot(data_, color=c, lw=2, label="{:.2f} hr" .format(t))
                s_title = "water elevation at y={} m" .format(y_trans)

        if plot_ground:
            ax.plot(grnd, 'k')

        ax.set_xlabel("x")
        # ax.set_ylabel(ylabel)
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0], 6])
        ax.set_xlim([0,np.shape(data_)[0]])
        ax.set_title(s_title)
        ax.legend()
        if fname!=None:
            # fn = "ytrans{}-t{}.png" .format(y_trans, t)
            plt.savefig(fname,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()


        if drawdomain:
            data_plot = self.read_data_xarray_gd()

            if fulldomain:
                figsize=(4,8)
            else:
                figsize=(8,6)


            fig, ax = plt.subplots(1,1, figsize=figsize)
            # --- new
            mask = (data_plot < -99999)
            masked_array = np.ma.array(data_plot, mask=mask)
            cmap = mpl.cm.BrBG_r
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(self.xgr, self.ygr, masked_array, vmin=-8.5, vmax=8.5, cmap=cmap)
            cnt = 0

            y = self.ygr[idy,0]
            ax.axhline(y=y, xmin=0, xmax=np.shape(data_plot)[1], color='k', lw=2)


            if fname!=None:
                fn = "{}-trns.png" .format(fname)
                plt.savefig(fn,
                            transparent=False, 
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            )
        
if __name__ == "__main__":

    pt = plot_transect("frun13-microdomain-1m-bldgs-3hr-tideloc1-tt2", var="H")
    pt.plot_water_level_transect(y_trans=450, 
                                 ts=[1.8], 
                                 h_plus_zs=True,
                                 drawdomain=False, 
                                 fulldomain=False,
                                 fname="temp.png")


    plt.show()

