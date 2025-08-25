import os
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.ndimage import label, generate_binary_structure
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class plot_wave_heights():
    """docstring for plot_wave_heights"""
    def __init__(self, var="H"):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_to_model = os.path.join(self.file_dir, "..", "..", "xbeach", "models")
        self.var = var

    def read_buildings(self, model_dir, rtn_mask=False):
        elev = self.read_data_xarray(model_dir, var="zb",t=0)
        # mask = (elev != 99)
        mask = (elev < 10)
        bldgs = np.ma.array(elev, mask=mask)
        if rtn_mask:
            return bldgs, mask
        return bldgs

    def save_max_wave_heights(self, model_runname, fn_out=None):
        fn = os.path.join(self.path_to_model, model_runname)
        data_save = self.read_data_xarray_max(fn, var="H")
        fn_out = os.path.join(self.file_dir, fn_out)
        np.save(fn_out, data_save)
        print("max wave height saved as: {}" .format(fn_out+".npy"))
    

    def plot_max_wave_height_bldg(self, model_runname, readlocal=False, vmax=None,
                                domain_size="estero", grey_background=False, fname=None):
        # read wave heights
        H = self.read_local_or_ncdf(model_runname, readlocal)
        
        # read buildings
        model_dir = os.path.join(self.path_to_model, model_runname)
        bldgs, mask = self.read_buildings(model_dir, rtn_mask=True)
        
        # assign max H to each building
        bldg_H = self.assign_max_to_bldgs(H, mask)

        xgr, ygr, zgr = self.read_grid(model_dir)

        if self.var == "H":
            cmap = mpl.cm.plasma
            if grey_background:
                cmap.set_bad("grey")
            else:
                cmap.set_bad(alpha=0)

            vmax = 1.0 if vmax == None else vmax
            vmin = 0
        else:
            cmap = mpl.cm.cividis
            cmap.set_bad('bisque')
            vmax = 3.0 if vmax == None else vmax
            vmin = 0.0


        if domain_size=="micro":
            figsize=(7,5)
        else:
            figsize=(3,8)
        
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.pcolormesh(xgr, ygr, zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r", zorder=0)
        pcm = ax.pcolormesh(xgr, ygr, bldg_H, vmin=0, vmax=1, cmap=cmap, zorder=1)
        plt.colorbar(pcm, ax=ax, extend="max", label="Max Wave Height (m)", aspect=40)
        ax.set_xlabel("x (m)")
        ax.set_xlabel("y (m)")

        self.savefig(fname)


    def assign_max_to_bldgs(self, data, mask):
        max_H = np.empty(np.shape(data))
        max_H[:] = np.nan
        labeled_mask, num_features = label(~mask)
        for i in range(num_features+1):
            if i == 0:
                continue
            m_ = labeled_mask==i
            m_ = ~m_
            m_ = np.pad(m_, pad_width=1, mode="constant", constant_values=True)

            shifted_up = m_[2:, 1:-1]
            shifted_down = m_[:-2, 1:-1]
            shifted_left = m_[1:-1, 2:]
            shifted_right = m_[1:-1, :-2]
            original_mask_trimmed = m_[1:-1, 1:-1]

            offset_mask = original_mask_trimmed & shifted_up & shifted_down & shifted_left & shifted_right
            offset_mask = ~offset_mask

            max_H[labeled_mask==i] = np.nanmax(data[offset_mask])

        return max_H

    def plot_max_wave_height(self, model_runname, readlocal=False, vmax=None, 
                             fname=None, prnt_read=False, single_frame=False,
                             plt_bldgs=True):
        
        data_plot = self.read_local_or_ncdf(model_runname, readlocal)

        # read in buildngs and grid
        model_runs = os.listdir(self.path_to_model)
        model_run = [i for i in model_runs if model_runname.split("max")[0] in i]
        model_run = [i for i in model_run if ".tar.gz" not in i][0]        
        model_dir = os.path.join(self.path_to_model, model_run)
        bldgs = self.read_buildings(model_dir)
        xgr, ygr, zgr = self.read_grid(model_dir)


        # fig, ax = plt.subplots(1,1, figsize=figsize)
        if single_frame:
            fig, ax0 = plt.subplots(1,1, figsize=(3,8))
        else:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16,9), gridspec_kw={'width_ratios': [1,2.8]})

        # setting up mask to ignore values less than 0
        mask = (zgr<=0)
        masked_array = np.ma.array(data_plot, mask=mask)
        
        # setting up colormap for water
        if self.var == "H":
            cmap = mpl.cm.plasma
            cmap.set_bad('grey')
            vmax = 1.0 if vmax == None else vmax
            vmin = 0
        else:
            cmap = mpl.cm.cividis
            cmap.set_bad('bisque')
            vmax = 3.0 if vmax == None else vmax
            vmin = 0.0

        
        # -- drawing first plot
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        if single_frame:
            ax_bar = ax0
        else:
            ax_bar = ax1
        plt.colorbar(pcm, ax=ax_bar, extend="max", label="Max Wave Height (m)", aspect=40)
        if plt_bldgs:
            cmap_bldg = mpl.cm.Greys_r
            cmap_bldg.set_bad(alpha=0)
            ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_xlabel("x (m)")
        ax0.set_ylabel("y (m)")
        # ax0.set_title(s)
        # self.remove_frame(ax0)

        if single_frame==False:
            # -- drawing second, zoomed in plot
            # full model domain
            box_lower_left = (2600, 5000)       # in world units
            dx, dy = 1000, 1000
            # continuing with zommed in plot
            box_upper_right = (box_lower_left[0]+dx, box_lower_left[1]+dy)

            id_ll = self.wrld_to_grid_index(xgr, ygr, box_lower_left)
            id_ur = self.wrld_to_grid_index(xgr, ygr, box_upper_right)
            
            xgr2 = xgr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            ygr2 = ygr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            masked_array2 = masked_array[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
            bldgs2 = bldgs[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]

            ax1.pcolormesh(xgr2, ygr2, masked_array2, vmin=vmin, vmax=vmax, cmap=cmap)
            ax1.pcolormesh(xgr2, ygr2, bldgs2, cmap=cmap_bldg)

            # # --old
            box_l = xgr2[0,-1] - xgr2[0,0]
            box_h = ygr2[-1,0] - ygr2[0,0]

            # # -- adding rectangle showing where zoomed in area is
            rect = patches.Rectangle(box_lower_left, box_l, box_h, linewidth=3, zorder=10, edgecolor='r', facecolor='none')
            ax0.add_patch(rect)

        # --- saving file
        self.savefig(fname)

    def plot_max_wave_height_stats(self, run1, run2, r1local=False, r2local=False, diff_colors=False, fname=None):
        # read max wave heights
        run1_max = self.read_local_or_ncdf(run1, r1local)
        run2_max = self.read_local_or_ncdf(run2, r2local)

        # read in zgrid - used to only consider overland values
        model_runs = os.listdir(self.path_to_model)
        model_run = [i for i in model_runs if run2.split("max")[0] in i]
        model_run = [i for i in model_run if ".tar.gz" not in i][0]        
        model_dir = os.path.join(self.path_to_model, model_run)
        _, _, zgr = self.read_grid(model_dir)

        # setting up mask; ignore all NaN's and cells that are considered water.
        mask = ~np.isnan(run1_max) & ~np.isnan(run2_max) & (zgr>=0)
        run1_max_nona, run2_max_nona = run1_max[mask], run2_max[mask]
        run1_max_nona = run1_max_nona.flatten()
        run2_max_nona = run2_max_nona.flatten()

        diff = run1_max_nona - run2_max_nona
        df = pd.DataFrame()
        df["run1"] = run1_max_nona
        df["run2"] = run2_max_nona
        df["diff"] = diff

        print(df['diff'].mean())
        df_less = df.loc[df["diff"]<=0]
        df_more = df.loc[df["diff"]>0]
        print(df_less["diff"].mean())
        print(df_more["diff"].mean())
        

        # plotting histogram
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        df["diff"].hist(ax=ax, 
                        bins=100, 
                        density=False, 
                        zorder=0, 
                        color="navajowhite", 
                        edgecolor='black', 
                        linewidth=0.3)
        ax.grid(False)
        ax.axvline(x=0,ymin=0, ymax=1, zorder=1, color="k", ls="-.", lw=1)
        r1_label = run1.split("max")[0]
        r2_label = run2.split("max")[0]
        ax.set_xlabel("Wave Height Difference (m)\n({} - {})" .format(r1_label, r2_label))
        ax.set_ylabel("Frequency")
        

        if diff_colors:
            # Iterate through the patches (bars) and set color conditionally
            for patch in ax.patches:
                if patch.get_x() < 0:
                    patch.set_facecolor('lightcoral')
                else:
                    patch.set_facecolor('cornflowerblue')
        self.savefig(fname)


    def plot_max_wave_height_scatter(self, run1, run2, r1local=False, r2local=False, fname=None):
        # read max wave heights
        run1_max = self.read_local_or_ncdf(run1, r1local)
        run2_max = self.read_local_or_ncdf(run2, r2local)

        # read in zgrid - used to only consider overland values
        model_runs = os.listdir(self.path_to_model)
        model_run = [i for i in model_runs if run2.split("max")[0] in i]
        model_run = [i for i in model_run if ".tar.gz" not in i][0]        
        model_dir = os.path.join(self.path_to_model, model_run)
        _, _, zgr = self.read_grid(model_dir)

        # setting up mask; ignore all NaN's and cells that are considered water.
        mask = ~np.isnan(run1_max) & ~np.isnan(run2_max) & (zgr>=0)
        run1_max_nona, run2_max_nona = run1_max[mask], run2_max[mask]
        run1_max_nona = run1_max_nona.flatten()
        run2_max_nona = run2_max_nona.flatten()

        fig, ax = plt.subplots(1,1, figsize=(5,4))
        ax.scatter(run1_max_nona, run2_max_nona, facecolor="none", edgecolor="cadetblue",lw=0.5, s=10, zorder=0)

        # regression to r^2 and best fit line
        slope, intercept, r_value, p_value, std_err = st.linregress(run1_max_nona, run2_max_nona)
        x = np.linspace(0,6, 100)
        y = slope*x + intercept
        ax.plot(x,y, ls="-", lw=1.5, color="purple", label="Regression")

        s1 = "Slope = {:0.4f}\n" .format(slope)
        s2 = "Intercept = {:0.4f}\n" .format(intercept)
        s3 = r"$r^2= $ {:0.4f}" .format(r_value)
        s = s1+s2+s3

        ax.text(x=0.95, y=0.05, s=s, 
                transform=ax.transAxes, 
                horizontalalignment='right', 
                verticalalignment="bottom",
                bbox=dict(facecolor='none', edgecolor='k'))

        # drawing 1 to 1 line
        ax.plot([-1,6], [-1,6], ls="-.", lw=1.0, zorder=1, color='k', label="1-to-1")

        ax.set_xlabel("{} ({})" .format("run2", "with bldgs"))
        ax.set_ylabel("{} ({})" .format("run7", "no bldgs"))
        ax.set_xlim([0,2])
        ax.set_ylim([0,2])
        ax.legend(loc="upper left")
        ax.set_title("Maximum Wave Height")

        self.savefig(fname)

    def savefig(self, fname):
        if fname!=None:
            plt.savefig(fname,
                        transparent=False, 
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        )
            plt.close()

    def plot_max_wave_height_diff(self, run1, run2, r1local=False, r2local=False, vmax=1, norm=False, fname=None):
        # get difference in wave heights
        run1_max = self.read_local_or_ncdf(run1, r1local)
        run2_max = self.read_local_or_ncdf(run2, r2local)

        mask = np.isnan(run1_max) & np.isnan(run2_max)
        run1_max = np.ma.array(run1_max, mask=mask) # here mask tells numpy which cells to ignore.
        run2_max = np.ma.array(run2_max, mask=mask) # here mask tells numpy which cells to ignore.

        if norm == True:
            # max_max = np.maximum(run1_max, run2_max)
            denom = (run1_max+run2_max)/2
            diff = ((run1_max - run2_max)/denom)
        else:
            diff = run1_max - run2_max

        # read in buildngs and grid
        model_runs = os.listdir(self.path_to_model)
        model_run = [i for i in model_runs if run2.split("max")[0] in i]
        model_run = [i for i in model_run if ".tar.gz" not in i][0]        
        model_dir = os.path.join(self.path_to_model, model_run)
        bldgs = self.read_buildings(model_dir)
        xgr, ygr, zgr = self.read_grid(model_dir)

        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16,9), gridspec_kw={'width_ratios': [1,2.8]})
        fig, ax0 = plt.subplots(1,1, figsize=(3,8))

        # determine where water is and setup mask to ignore those cells.
        mask = (zgr<0)
        masked_array = np.ma.array(diff, mask=mask) # here mask tells numpy which cells to ignore.
        
        # setting up colormap for water
        if self.var == "H":
            cmap = mpl.cm.RdBu
            cmap.set_bad('grey')
            vmax = 1.0 if vmax == None else vmax
            vmin = -vmax
        else:
            cmap = mpl.cm.cividis
            cmap.set_bad('bisque')
            vmax = 3.0 if vmax == None else vmax
            vmin = 0.0

        cmap_bldg = mpl.cm.Greys_r
        cmap_bldg.set_bad(alpha=0)

        # -- drawing first plot
        if norm:
            label = "Percent Difference"
            vmax = 1
            vmin = -1
            extend = None
        else:
            label = "Difference (m) (run2 - run7)"
            extend = "both"
        pcm = ax0.pcolormesh(xgr, ygr, masked_array, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(pcm, ax=ax0, extend=extend, label=label, aspect=40)
        ax0.pcolormesh(xgr, ygr, bldgs, cmap=cmap_bldg)
        ax0.set_xlabel("x (m)")
        ax0.set_ylabel("y (m)")
        # # -- drawing second, zoomed in plot
        # # full model domain
        # box_lower_left = (2600, 5000)       # in world units
        # dx, dy = 1000, 1000
        # # continuing with zommed in plot
        # box_upper_right = (box_lower_left[0]+dx, box_lower_left[1]+dy)

        # id_ll = self.wrld_to_grid_index(xgr, ygr, box_lower_left)
        # id_ur = self.wrld_to_grid_index(xgr, ygr, box_upper_right)
        
        # xgr2 = xgr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        # ygr2 = ygr[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        # masked_array2 = masked_array[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]
        # bldgs2 = bldgs[id_ll[1]:id_ur[1], id_ll[0]:id_ur[0]]

        # ax1.pcolormesh(xgr2, ygr2, masked_array2, vmin=vmin, vmax=vmax, cmap=cmap)
        # ax1.pcolormesh(xgr2, ygr2, bldgs2, cmap=cmap_bldg)

        # # # --old
        # box_l = xgr2[0,-1] - xgr2[0,0]
        # box_h = ygr2[-1,0] - ygr2[0,0]

        # # # -- adding rectangle showing where zoomed in area is
        # rect = patches.Rectangle(box_lower_left, box_l, box_h, linewidth=3, zorder=10, edgecolor='r', facecolor='none')
        # ax0.add_patch(rect)

        # --- saving file
        self.savefig(fname)

    def read_local_or_ncdf(self, run, rlocal):
        if rlocal:
            fn = os.path.join(self.file_dir, run)
            rmax = np.load(fn)
        else:
            fn = os.path.join(self.path_to_model, run)
            rmax = self.read_data_xarray_max(fn, var="H")
        return rmax

    def wrld_to_grid_index(self, xgr, ygr, xy):
        idx = np.argmin(np.abs(xgr[0,:] - xy[0]))
        idy = np.argmin(np.abs(ygr[:,0] - xy[1]))        
        return (idx,idy)

    def remove_frame(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    def read_data_xarray_max(self, model_dir, var, prnt_read=False):
        fn = os.path.join(model_dir, "xboutput.nc")

        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        max_vals = ds[var].max(dim="globaltime").values[:,:]
        return max_vals
    
    def read_data_xarray(self, model_dir, var, t, prnt_read=False, rtn_time_array=False):
        fn = os.path.join(model_dir, "xboutput.nc")
        ds = xr.open_dataset(fn, chunks={"globaltime": 100})
        if prnt_read:
            print("Dataset object read:")
            print(ds)
            print("\n\n")
        
        slice_data = ds[var].isel(globaltime=slice(t,t+1))
        if rtn_time_array:
            time = ds["globaltime"].values
            # print("Last time step: {} hr." .format(time[-1]/60/60))
            return slice_data.values[0,:,:], time
        else:
            return slice_data.values[0,:,:]
    
    def read_grid(self, model_dir):
        fn_xgrid = os.path.join(model_dir, "x.grd")
        if os.path.isfile(fn_xgrid):
            xgrid = os.path.join(model_dir, "x.grd")
            ygrid = os.path.join(model_dir, "y.grd")
            zgrid = os.path.join(model_dir, "z.grd")

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
            
            zgr = np.zeros((len(ys), len(xs)))
            with open(zgrid,'r') as f:
                for cnt, line in enumerate(f.readlines()):
                    z_ = [float(i.strip()) for i in line.split()]
                    zgr[cnt,:] = z_

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

        return xgr, ygr, zgr


if __name__ == "__main__":
    pwh = plot_wave_heights(var="H")
    pwh.plot_max_wave_height_bldg(
                                  model_runname="frun13-microdomain-1m-bldgs-3hr-tideloc1-tt2",
                                  readlocal=False,
                                  vmax=1,
                                  domain_size="micro",  # or estero
                                  # fname="temp.png"
                                )

    # pwh.plot_max_wave_height(model_runname="run7max.npy", 
    #                          readlocal=True, 
    #                          vmax=1, 
    #                          single_frame=True, 
    #                          plt_bldgs=False,
    #                          fname="run7max.png"
    #                          )

    # pwh.plot_max_wave_height_diff(run1="run2max.npy", 
    #                               run2="run7max.npy", 
    #                               r1local=True, 
    #                               r2local=True, 
    #                               vmax=1, 
    #                               norm=False,
    #                               fname="diff_r2r7_map.png")

    # pwh.plot_max_wave_height_scatter(run1="run2max.npy", 
    #                                  run2="run7max.npy", 
    #                                  r1local=True, 
    #                                  r2local=True,
    #                                  fname=None,
    #                                  # fname="scatter.png"
    #                                  )
    # pwh.plot_max_wave_height_stats(run1="run2max.npy", 
    #                                run2="run7max.npy", 
    #                                r1local=True, 
    #                                r2local=True,
    #                                diff_colors=True,
    #                                # fname=None,
    #                                fname="r2r7_hist.png"
    #                                )


    plt.show()

