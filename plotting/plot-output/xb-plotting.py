import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import imageio
import netCDF4
import seaborn as sns


class xb_plotting():
    def __init__(self, model_runname, var, t=1):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_runname = model_runname
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "models", self.model_runname)

        self.t = t
        self.data, self.time = self.read_data(var)
        self.data_zs0, _ = self.read_data("zs0")

        self.xgr, self.ygr = self.read_grid()
        self.var = var
        
        if self.model_runname == "test-waves11":
            self.time = np.linspace(self.time[0], self.time[-1], len(self.time))
        self.read_buildings()

    def read_buildings(self):
        elev, _ = self.read_data("zb")
        elev = elev[0,:,:]
        mask = (elev != 99)
        self.bldgs = np.ma.array(elev, mask=mask)

    def read_data(self, var, timekey="globaltime"):
        if var == None:
            raise ValueError("var not identified")

        # reading/processing data
        fn = os.path.join(self.path_to_model, "xboutput.nc")
        data, time = self.read_h5(fn, var, timekey)
        return data, time

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

    def read_h5(self, fn, var=None, timekey="globaltime"):
        if var == None:
            raise ValueError("var not identified")
        nc = netCDF4.Dataset(fn, mode="r")            # reading the nc file and creating Dataset
        data = np.asarray(nc[var])
        time = np.asarray(nc[timekey])
        print("Output data, {}, read with shape: {}" .format(var, np.shape(data)))
        
        return data, time

    def plot_buildings(self, t=None, fname=None, vmax=None):
        if t!=None:
            self.t = t
        data_plot = self.data[self.t,:,:]
        data_shape = np.shape(data_plot)
        if data_shape[1]>data_shape[0]:
            figsize = (8,1)
            txt_x = 0.02
            txt_y = 0.3
        else:
            figsize = (4,8)
            txt_x = 0.1
            txt_y = 0.83

        # creating figure
        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,5]})
        
        # filtering out nan data
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
            vmin = 1.0

        # setting up colormap for buildings
        cmap_bldg = mpl.cm.Greys_r
        cmap_bldg.set_bad(alpha=0)
        if vmax==None:
            vmax = np.nanmax(self.data) # - self.data_zs0)
        
        # -- drawing first plot
        pcm = ax0.pcolormesh(self.xgr, self.ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax1, extend="max")

        # s = "Time: {:8.0f} s, {:11.0f} h" .format(self.time[t], self.time[t]/3600)
        if self.var == "H":
            s = "Wave Height (m)\nTime: {:2.0f}h ({:8.0f}s)" .format(self.time[t]/3600, self.time[t])
        elif self.var == "zs":
            s = "Water Elevation (m)\nTime: {:2.0f}h ({:8.0f}s)" .format(self.time[t]/3600, self.time[t])
        

        ax0.pcolormesh(self.xgr, self.ygr, self.bldgs, cmap=cmap_bldg)
        ax0.set_title(s)
        

        # -- drawing second, zoomed in plot
        box_start = 2800
        idx = np.argmin(np.abs(self.xgr[0,:] - box_start))
        xgr2, ygr2, masked_array2, bldgs2 = self.xgr[:,idx:], self.ygr[:,idx:], masked_array[:,idx:], self.bldgs[:,idx:]
        ax1.pcolormesh(xgr2, ygr2, masked_array2, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.pcolormesh(xgr2, ygr2, bldgs2, cmap=cmap_bldg)

        box_l = self.xgr[0,-1] - box_start
        box_h = self.ygr[-1,0] - self.ygr[0,0] - 2

        # -- adding rectangle showing where zoomed in area is
        rect = patches.Rectangle((box_start, self.ygr[0,0]+1), box_l, box_h, linewidth=3, zorder=10, edgecolor='r', facecolor='none')
        ax0.add_patch(rect)
        
        
        # --- old
        if fname != None:
            plt.savefig(fname,
                        transparent=False, 
                        dpi=150,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()

    def plot(self, t=None, fname=None, vmax=None):
        if t!=None:
            self.t = t
        data_plot = self.data[self.t,:,:]   # - self.data_zs0[self.t,:,:]
        # data_plot = np.nan_to_num(zs_plot)
        # data_plot[data_plot<-99999] = 0
        data_shape = np.shape(data_plot)
        if data_shape[1]>data_shape[0]:
            figsize = (8,1)
            txt_x = 0.02
            txt_y = 0.3
        else:
            figsize = (4,8)
            txt_x = 0.1
            txt_y = 0.83
        fig, ax = plt.subplots(1,1, figsize=figsize)
        # --- new
        mask = (data_plot < -99999)
        masked_array = np.ma.array(data_plot, mask=mask)
        # cmap = mpl.cm.viridis
        cmap = mpl.cm.Blues
        cmap.set_bad('bisque')
        if vmax==None:
            vmax = np.nanmax(self.data) # - self.data_zs0)
            # vmax = 4

        ax.pcolormesh(self.xgr, self.ygr, masked_array, vmin=-0.5, vmax=vmax, cmap=cmap)

        # ax.pcolormesh(self.xgr, self.ygr, masked_array, vmin=-0.5, vmax=1, cmap=cmap)

        s = "Time\n{:8.0f} s\n{:11.0f} h" .format(self.time[t], self.time[t]/3600)
        ax.text(x=txt_x, y=txt_y, s=s, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.9))


        # --- old
        if fname != None:
            plt.savefig(fname,
                        transparent=False, 
                        dpi=100,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()

    def make_animation_buildings(self, tstart=None, tstop=None, makefigs=True, vmax=None):

        if tstart == None:
            tstart_idx = 0
        else:
            tstart_idx = np.argmin(np.abs(self.time - tstart))
        if tstop == None:
            tstop_idx = np.shape(self.data)[0]
        else:
            tstop_idx = np.argmin(np.abs(self.time - tstop))             

        
        # reading data
        data_plot = self.data[tstart_idx:tstop_idx,:,:]

        # --- making images to comprise video
        temp_dir = os.path.join(self.file_dir, "temp")
        if makefigs:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            self.make_directory(temp_dir)

            for t in range(tstart_idx, tstop_idx):
                if t%100==0:
                    print(self.time[t]/3600)
                fn = os.path.join(temp_dir, "f{}.png" .format(t))
                self.plot_buildings(t, fname=fn, vmax=vmax)
                plt.close()
        
        # --- making video
        video_name = '{}-{}.mp4' .format(self.model_runname, self.var)
        print(video_name)
        writer = imageio.get_writer(video_name, fps=10)
        for step in range(tstart_idx, tstop_idx):
            fn = os.path.join(temp_dir, "f{}.png" .format(step))
            if os.path.isfile(fn):
                writer.append_data(imageio.imread(fn))
        writer.close()

    def make_animation_imageio(self, tstart=None, tstop=None, makefigs=True):
        if tstart == None:
            tstart = 0
        if tstop == None:
            tstop = np.shape(self.data)[0]
        
        # reading data
        data_plot = self.data[tstart:tstop,:,:]

        # --- making images to comprise video
        temp_dir = os.path.join(self.file_dir, "temp")
        if makefigs:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
            self.make_directory(temp_dir)

            for t in range(tstart, tstop):
                if t%100==0:
                    print(t)
                fn = os.path.join(temp_dir, "f{}.png" .format(t))
                self.plot(t, fname=fn, vmax=np.nanmax(data_plot))
                plt.close()
        
        # --- making video
        video_name = '{}-{}.mp4' .format(self.model_runname, self.var)
        writer = imageio.get_writer(video_name, fps=10)
        print(np.shape(data_plot[0]))
        for step in range(tstart, tstop):
            fn = os.path.join(temp_dir, "f{}.png" .format(step))
            if os.path.isfile(fn):
                writer.append_data(imageio.imread(fn))
        writer.close()

    def plot_water_level_point(self, xys, drawdomain=False, savefig=False):
        colors = sns.color_palette("husl")
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        cnt = 0

        for xy in xys:
            idx = np.argmin(np.abs(self.xgr[0,:] - xy[0]))
            idy = np.argmin(np.abs(self.ygr[:,0] - xy[1]))
            data_ = self.data[:, idy, idx] #- self.data_zs0[:, idy, idx]


            # -- plotting
            data_[data_<-99999] = 0

            ax.plot(self.time/3600, data_, label="{}" .format(cnt), color=colors[cnt], lw=2)
            cnt += 1

        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Elevation (m)")
        ax.legend()
        if savefig:
            fn = "elevation-timeseries.png"
            plt.savefig(fn,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )

        if drawdomain:
            data_plot = self.data[0,:,:]

            fig, ax = plt.subplots(1,1, figsize=(4,8))
            # --- new
            mask = (data_plot < -99999)
            masked_array = np.ma.array(data_plot, mask=mask)
            cmap = mpl.cm.Blues
            cmap.set_bad('bisque',1.)
            ax.pcolormesh(self.xgr, self.ygr, masked_array, vmin=-0.5, vmax=np.max(self.data), cmap=cmap)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            cnt = 0

            for xy in xys:
                idx = np.argmin(np.abs(self.xgr[0,:] - xy[0]))
                idy = np.argmin(np.abs(self.ygr[:,0] - xy[1]))
                # -- new
                x, y = self.xgr[0,idx], self.ygr[idy, 0]                

                # -- old
                # x, y = self.xgr[0,xy[0]], self.ygr[xy[1],0]
                ax.scatter(x, y, color=colors[cnt],s=50)
                ax.annotate("{}" .format(cnt), (x, y))

                # ax.scatter(xy[0], xy[1], color=colors[cnt],s=50)
                # ax.annotate("{}" .format(cnt), (xy[0], xy[1]))
                cnt += 1

            if savefig:
                fn = "obs-points.png"
                plt.savefig(fn,
                            transparent=False, 
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            )

    def var2label(self, var):
        v2l = { "el":"Water Elevation",
                "hs": "Significant Wave Height",
                "Tp": "Peak Period"
        }
        v2y = { "el": "Water Elevation (m; ___)",
                "hs": "Significant Wave Height (m)",
                "Tp": "Peak Period (s)"
        }
        
        c = {"el": 0, "hs": 1, "Tp": 2}
        colors = sns.color_palette("crest", n_colors=len(c.keys()))
        color = colors[c[var]]

        return v2l[var], v2y[var], color


    def plot_water_level_transect(self, y_trans, ts, plot_trans=True, drawdomain=False, savefig=False):
        fig, ax = plt.subplots(1,1,figsize=(10,4))
        colors = sns.color_palette("viridis")
        if plot_trans == True:
            trns, _ = self.read_data("zb")

        # get data for variable
        cnt = 0
        for t in ts:
            idy = np.argmin(np.abs(self.ygr[:,0] - y_trans))
            data_ = self.data[t,idy,:]
            data_[data_<-99999] = 0
            _, ylabel, c = self.var2label("el")
            c = colors[cnt]

            ax.plot(data_, color=c, lw=2)
            if plot_trans:
                trns_ = trns[t, idy,:]
                ax.plot(trns_, 'k')
            cnt += 1

        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.set_xlim([0,np.shape(data_)[0]])
        ax.set_title("water elevation at transect: {}" .format(y_trans))
        if plot_trans:
            ylim = ax.get_ylim()
            ax.set_ylim([ylim[0], 5])


        if savefig == True:
            fn = "ytrans{}-t{}.png" .format(y_trans, t)
            plt.savefig(fn,
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
            # for xy in xys:
            #     ax.scatter(xy[0], xy[1], color=colors[cnt],s=50)
            #     ax.annotate("{}" .format(cnt), (xy[0], xy[1]))
            #     cnt += 1

            if savefig:
                fn = "obs-trns.png"
                plt.savefig(fn,
                            transparent=False, 
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1,
                            )


    def compare_forcing2output(self, xb_loc, drawdomain=False, savefig=False):
        if xb_loc == "lower-left":
            xidx = 1
            yidx = 1
            frcng_dat = "xbeach1.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"
                    
        elif xb_loc == "upper-left":
            xidx = 1
            yidx = np.shape(self.data)[1]-1
            frcng_dat = "xbeach4.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"
        
        elif xb_loc == "upper-right":
            xidx = np.shape(self.data)[2]-1
            yidx = np.shape(self.data)[1]-1
            frcng_dat = "xbeach3.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"

        elif xb_loc == "lower-right":
            xidx = np.shape(self.data)[2]-1
            yidx = 0
            frcng_dat = "xbeach2.dat"
            xb_label = "Model Output"
            fr_label = "Forcing Input"

        elif xb_loc == "adcirc-offshore":
            xidx = np.argmin(np.abs(self.xgr[0,:] - 2585))
            yidx = np.argmin(np.abs(self.ygr[:,0] - 5640))
            frcng_dat = "xbeach5.dat"
            xb_label = "XBeach"
            fr_label = "ADCIRC/SWAN"

        elif xb_loc == "adcirc-onshore":
            xidx = np.argmin(np.abs(self.xgr[0,:] - 3069))
            yidx = np.argmin(np.abs(self.ygr[:,0] - 5625))
            frcng_dat = "xbeach6.dat"
            xb_label = "XBeach"
            fr_label = "ADCIRC/SWAN"

        else:
            raise ValueError("Incorrect xb_loc provided.")    

        output = self.data[:,yidx, xidx]
        output[output<-99999] = 0

        fn = os.path.join(self.file_dir, "..", "..", "..", "data", "forcing", frcng_dat)
        df = self.frcing_to_dataframe(fn)

        fig, ax = plt.subplots(1,1, figsize=(8,3))
        ax.plot(self.time/3600, output, color="dodgerblue", lw=2.5, label=xb_label)
        ax.plot(df["t_sec"]/3600, df["el"], color="k", lw=1.5, label=fr_label)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Water Elevation (m)")
        ax.set_xlim([0,45])
        ax.legend()

        if savefig == True:
            fn = "frcng2output.png"
            plt.savefig(fn,
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
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            
            cnt = 0
            colors = sns.color_palette("husl")
            x, y = self.xgr[0,xidx], self.ygr[yidx, 0]                
            ax.scatter(x, y, color=colors[cnt],s=50)
            ax.annotate("{}" .format(cnt), (x, y))

        if savefig == True:
            fn = "frcng2output_domain.png"
            plt.savefig(fn,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()


    def frcing_to_dataframe(self, fn, n_header=3):
        t, el, wx, wy, hs, Tp, wavedir = [], [], [], [], [], [], [],
        with open(fn,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                if cnt < n_header:
                    if "VARIABLES" in line:
                        var = [x.strip() for x in line.split()]
                        var = [i for i in var if i!="VARIABLES"]
                        var = [i for i in var if i!="="]
                    continue
                t_, el_, wx_, wy_, hs_, Tp_, wavedir_ = [float(x.strip()) for x in line.split()]
                t.append(t_)
                el.append(el_)
                wx.append(wx_)
                wy.append(wy_)
                hs.append(hs_)
                Tp.append(Tp_)
                wavedir.append(wavedir_)
        

        # TODO confirm unit conversions with Don
        df = pd.DataFrame()
        df["t"] = t
        df["el"] = el
        df["wx"] = wx
        df["wy"] = wy
        df["hs"] = hs
        df["Tp"] = Tp
        df["wavedir"] = wavedir

        df["el"] = df["el"]*0.3048
        df["hs"] = df["hs"]*0.3048

        dt = (df["t"].iloc[1] - df["t"].iloc[0])*60*60         # time setp in seconds; converting from hours.
        df["t_sec"] = np.linspace(0, (len(df)-1)*dt, len(df))
        df["t_hr"] = df["t_sec"]/3600

        return df
        
    def make_directory(self, path_out):
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        return path_out




if __name__ == "__main__":

    # --
    xbp = xb_plotting(model_runname="test-waves22", var="H")
    
    # xbp.make_animation_imageio(tstart=1200, tstop=1800, makefigs=True)
    # xbp.make_animation_imageio(tstart=1100, tstop=None, makefigs=True)
    # xbp.plot(t=1600)

    xbp.make_animation_buildings(tstart=22*3600, tstop=85700, makefigs=True)
    xbp.plot_buildings(t=2850, vmax=0.5)

    # xbp.plot_water_level_point(xys=[(0,5570), (2597,5570), (3073,5570)], drawdomain=True, savefig=False)
    # xbp.plot_water_level_point(xys=[(600, 250)], drawdomain=True, savefig=False)

    # xbp.compare_forcing2output(xb_loc="lower-left", drawdomain=False, savefig=False)

    # xbp.plot_water_level_transect(y_trans=40, ts=[-1], drawdomain=True, plot_trans=True, savefig=False)
    plt.show()


