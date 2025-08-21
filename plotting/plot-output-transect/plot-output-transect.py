import os
import numpy as np
import xarray as xr
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
        
        # slice_data = ds[var].isel(globaltime=slice(t,t+1))
        slice_data = ds[var][t,idy,:]

        if rtn_time_array:
            time = ds["globaltime"].values
            return slice_data.values, time
        else:
            return slice_data.values


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


    def plot_water_level_transect(self, y_trans, ts, plot_trans=True, drawdomain=False, fname=None):
        fig, ax = plt.subplots(1,1,figsize=(10,4))
        colors = sns.color_palette("viridis")
        if plot_trans == True:
            trns = self.read_data_xarray(var="zb", t=0)

        # get data for variable
        cnt = 0
        for t in ts:
            idy = np.argmin(np.abs(self.ygr[:,0] - y_trans))
            data = self.read_data_xarray(var=self.var, t=t)
            data_ = data[idy,:]
            data_[data_<-99999] = 0
            # _, ylabel, c = self.var2label("el")
            c = colors[cnt]

            ax.plot(data_, color=c, lw=2)
            if plot_trans:
                trns_ = trns[idy,:]
                ax.plot(trns_, 'k')
            cnt += 1

        ax.set_xlabel("x")
        # ax.set_ylabel(ylabel)
        ax.set_xlim([0,np.shape(data_)[0]])
        ax.set_title("water elevation at y={} m" .format(y_trans))
        if plot_bed:
            ylim = ax.get_ylim()
            ax.set_ylim([ylim[0], 5])


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
            data_plot = self.data[0,:,:]

            fig, ax = plt.subplots(1,1, figsize=(4,8))
            # --- new
            mask = (data_plot < -99999)
            masked_array = np.ma.array(data_plot, mask=mask)
            cmap = mpl.cm.Blues
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(self.xgr, self.ygr, masked_array, vmin=-0.5, vmax=np.max(self.data), cmap=cmap)
            cnt = 0

            y = self.ygr[y_trans,0]
            ax.axhline(y=y, xmin=0, xmax=np.shape(data_plot)[1], color='k', lw=2)


            if savefig:
                fn = "obs-trns.png"
                plt.savefig(fn,
                            transparent=False, 
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            )
        
if __name__ == "__main__":

    pt = plot_transect("run6-5m-bldgs-3hr-tideloc4", var="zs")
    pt.plot_water_level_transect(y_trans=500, ts=[100], fname="temp.png")

    plt.show()

    plt.show()