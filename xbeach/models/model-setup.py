import os
import sys
import numpy as np
from datetime import datetime
from scipy import interpolate
import matplotlib.pyplot as plt

import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.warp import calculate_default_transform, reproject, Resampling

import xbTools
from xbTools.grid.creation import xgrid, ygrid
from xbTools.grid.extension import seaward_extend, lateral_extend
from xbTools.xbeachtools import XBeachModelSetup
from xbTools.general import wave_functions, visualize_mesh

class setup_xbeach():
    def __init__(self):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.setup_inputs()
        self.set_xbeach_params()

    def input_vals(self):
        inputs = {
        "model_name": "test",
        "path_to_dem": os.path.join(self.file_dir, "..", "..", "data", "dem", "dem-resampled.tiff"),
        "path_to_domain": os.path.join(self.file_dir, "..", "..", "data", "xbeach-domain", "xbeach-domain-epsg32617.geojson"),
        # "path_to_buildings": None,
        "path_to_buildings": os.path.join(self.file_dir, "..", "..", "data", "buildings", "ft_myers_bldgs.geojson"),
        "path_to_forcing": os.path.join(self.file_dir, "..", "..", "data", "forcing"),
        "forcing_files": ["xbeach1.dat", "xbeach2.dat", "xbeach3.dat", "xbeach4.dat"],
        "local_utm_epsg": "EPSG:32617",
        "drawfigs": True,
        "savefigs": True,
        }
        return inputs

    def set_xbeach_params(self):
        self.xbeach_params = {
                    "xbeach_res" : 30,          # in local utm units (m here).
                    
                    # -- grid input --
                    # "xori"      : 0,            # x-coordinate of origin of axis
                    # "yori"      : 0,            # y-coordinate of origin of axis
                    "depfile"   : "z.grd",      # Name of the input bathymetry file
                    "vardx"     : 0,            # Switch for variable grid spacing
                    "xfile"     : "x.grd",      # Name of the file containing x-coordinates of the calculation grid
                    "yfile"     : "y.grd",      # Name of the file containing y-coordinates of the calculation grid
                    "posdwn"    : -1,           # Bathymetry is specified positive down (1) or positive up (-1)
                    "thetamax"  : 90,           # Higher directional limit (angle w.r.t computational x-axis)   
                    "thetamin"  : -90,          # Lower directional limit (angle w.r.t computational x-axis)
                    # "single_dir": 0,            # Turn on stationary model for refraction, surfbeat based on mean direction
                    "dtheta"    : 10,           # Directional resolution; 
                    "dtheta_s"  : 10,           # Directional in case of stationary refraction; not used in stationary mode
                    
                    # -- numerics input --
                    "CFL"       : 0.4,          # Maximum courant-friedrichs-lewy number
                    # "eps"       : 0.001,        # Threshold water depth above which cells are considered wet
                    # "front"     : "wlevel",     # Switch for seaward flow boundary (abs_1d, abs_2d, wall, wlevel, nonh_1d, waveflume); switches to abs_1d for
                    # "back"      : "wlevel",     # Switch for boundary at bay side (wall, abs_1d, abs_2d, wlevel)   
                    # "scheme"    : "warmbeam",   # Numerical scheme for wave propagation (upwind_1, lax_wendroff, upwind_2, warmbeam)
                    # "left"      : "neumann",    # Switch for lateral boundary at ny+1 (neumann, wall, no_advec, neumann_v, abs_1d)
                    # "right"     : "neumann",    # Switch for lateral boundary at 0    (neumann, wall, no_advec, neumann_v, abs_1d)
                    # "maxdtfac"     : 500,    # Maximum increase/decrease in time stp in explosion prevention mechanism


                    # -- time input --
                    "tstart"    : 0,            # Start time of output, in morphological time
                    "tintg"     : 60,           # interval time of global output
                    "tintm"     : 400,          # interval time of mean, var, max, min output
                    "tintp"     : 60,           # interval time of point/runup gauge output
                    "tstop"     : 200000,       # end time seconds
                    "taper"     : 200,          # Spin-up time of wave boundary conditions, in morphological time
                    # "dtset"     : 0.5,          # Fixed timestep, overrides use of cfl

                    # -- general constants --
                    # "rho"   : 1025,             # Density of water
                    # "g"     : 9.81,             # Gravitational acceleration

                    # -- boundary conditions --
                    "zs0file"   : "water_elev.dat",     # Name of tide boundary condition series
                    "tideloc"   : 4,            # Number of corner points on which a tide time series is specified
                    # "tidetype"  : "velocity",    # Switch for offfshore boundary, velocity boundary or instant water level boundary (instant, velocity, hybrid; default velocity)
                    "zs0"       : 0,            # Inital water level
                    # "paulrevere": 0 ,         # Specifies tide on sea and land or two sea points if tideloc = 2 (land, sea)
                    # "tidelen"   : None,       # length of tide signal (doesn't appear to be read in xbeach)
                    # "bcfile"    : "jonswap.txt",         # Name of spectrum file
                    "bcfile"    : "loclist.txt",         # Name of spectrum file; use if providing multiple spectra (nspectrumloc>1)


                    # -- wave calculation options
                    "wavemodel"   : "surfbeat",   # stationary (0), surfbeat (1) or non-hydrostatic (2)
                    "wbctype"     : "jonstable",    # New wave boundary condition type
                    "nspectrumloc": 2,          # number of wave spectra in offshore boundary
                    # "wbcversion"  : 3,              # wave boundary condition version
                    # "instat"      : "jons",
                    # "break"     : 1,
                    # "wci"       : 0,
                    # "roller"    : 1,
                    # "beta"      : 0.1,
                    # "gamma"     : 0.52,
                    # "gammax"    : 3,
                    # "alpha"     : 1,
                    # "delta"     : 0.0 ,
                    # "n"         : 10,
                    # "maxerror"  : 0.001,        # Maximum wave height error in wave stationary iteration (default=0.0001)
                    # "maxiter"   : 1000,          # Maximum number of iterations in wave stationary (default=500)


                    # -- Flow calculation options --
                    # "nuh"      : 0.1,
                    # "nuhfac"   : 1,
                    # "nuhv"     : 1,
                    # "umin"     : 0.0,

                    # -- sediment transport options --
                    "sedtrans": 0,      # Turn on sediment transport
                    # "dico"  : 1,
                    # "D50"   : 0.0002,
                    # "D90"   : 0.0003,
                    # "rhos"  : 2650,
                    # "z0"    : 0.006,

                    # -- morphologic opttions --
                    "morphology": 0,            # Turn on morphology
                    "struct"    : 0,            # turn on hard structures (1) or not (0)
                    # "ne_layer"  : "ne_layer.grd", # filename for non-erodible layer
                    # "morfac"   : 0,
                    # "morstart" : 3800,
                    # "por"      : 0.4,
                    # "dryslp"   : 1.0,
                    # "wetslp"   : 0.3 ,
                    # "hswitch"  : 0.1,

                    # -- output options --
                    "global_var": ["zs", "zs0", "zs1", "H", "hh", "zb"]
                    }
    
    def setup_inputs(self):
        input_vals = self.input_vals()
        self.set_model_name(input_vals["model_name"])
        self.set_path_to_dem(input_vals["path_to_dem"])
        self.set_path_to_domain(input_vals["path_to_domain"])
        self.set_path_to_buildings(input_vals["path_to_buildings"])

        self.set_path_to_forcing(input_vals["path_to_forcing"])
        self.set_forcing_files(input_vals["forcing_files"])
        self.set_local_utm_epsg(input_vals["local_utm_epsg"])
        self.set_drawfigs(input_vals["drawfigs"])
        self.set_savefigs(input_vals["savefigs"])

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.path_to_model = os.path.join(self.file_dir, model_name)
        self.make_directory(self.path_to_model)

    def set_path_to_dem(self, val=None):
        self.path_to_dem = val

    def set_path_to_domain(self, val=None):
        self.path_to_domain = val

    def set_path_to_buildings(self, val=None):
        self.path_to_buildings = val
    
    def set_path_to_forcing(self, val=None):
        self.path_to_forcing = val
    
    def set_forcing_files(self, val=None):
        self.forcing_files = val

    def set_local_utm_epsg(self, val=None):
        self.local_epsg = val

    def set_drawfigs(self, val=None):
        self.drawfigs = val

    def set_savefigs(self, val=None):
        self.path_to_figs = os.path.join(self.path_to_model, "figs")
        self.make_directory(self.path_to_figs)
        self.savefig = val

    # ==========================================================================
    def setup_model(self):
        grid_df, bathy, x, y = self.raster_to_xbeach_grid()         # from raster to rotated xbeach grid.
        xgr, ygr, zgr = self.xbtools_grid(bathy, x, y)              # using xbeach tools to prepare xbeach grid
        zgr, nesgr = self.add_buildings(xgr, ygr, zgr, grid_df)
        frcng_df = self.setup_forcing()
        self.create_model(xgr, ygr, zgr, nesgr, frcng_df)                          # writing out xbeach model

    def raster_to_xbeach_grid(self):
        """
        reads raster data, converts to local crs (local crs units must be m.),
        crops raster to xbeach-domain, setsup an empty rotated grid, then 
        cell-by-cell populates the empty grid with the dem elevations.
        Results in an xbeach grid that is orientied with (0,0) in the lower 
        left-hand corner. 
        """
        gdf_domain = gpd.read_file(self.path_to_domain)     # read in xbeach domain
        gdf_domain.to_crs(self.local_epsg, inplace=True)

        self.reproject_raster()                             # reproject raster to local crs
        grid_df, grid, x, y = self.setup_grid(gdf_domain)   # setup xbeach grid with (0,0) in lower left corner. grid is rotated.
        os.remove("temp.tiff")
        if self.drawfigs:
            fig, ax = plt.subplots(1,1, figsize=(5,8))
            ax.imshow(grid.T, origin="lower", vmin=-8.5, vmax=8.5, cmap="BrBG_r")
            ax.set_xlabel("x")
            ax.set_ylabel("y", rotation=0)
            ax.set_title("r2xg")
            
            if self.savefig:
                fn = os.path.join(self.path_to_figs, "xbeach-grid.png")
                plt.savefig(fn, 
                            transparent=False, 
                            dpi=500,
                            bbox_inches="tight",
                            pad_inches=0.1,
                            )

        return grid_df, grid, x, y
    
    def xbtools_grid(self, bathy, x, y, zgrid_row=None):
        if zgrid_row==None:
            zgrid_row = int(np.rint(np.shape(bathy)[1]/2))
        
        if self.xbeach_params["vardx"]==1:
            xgr,zgr = xgrid(
                            x=x, 
                            z=bathy[:,zgrid_row], 
                            dxmin=self.xbeach_params["xbeach_res"],       # minimum grid resolution (default=5)
                            vardx=self.xbeach_params["vardx"],            # spatially varying grid (1) or constant (0) (default=1)
                            )

            ygr = ygrid(y=y,
                        dymin=self.xbeach_params["xbeach_res"],      # (default=5)
                        )
        else:
            xgr, ygr = x, y

        # # --- original interpolation
        # interp = interpolate.RegularGridInterpolator((x,y),bathy, method="cubic", bounds_error=False, fill_value=None)
        # pts = np.meshgrid(xgr,ygr)
        # xgr, ygr = np.meshgrid(xgr, ygr)
        # zgr = interp(pts)
        # # --- 
        
        # --- new
        pts = np.meshgrid(xgr,ygr)
        zgr = interpolate.interpn((x,y), bathy, pts, bounds_error=False, fill_value=None, method="pchip")
        xgr, ygr = np.meshgrid(xgr, ygr)
        # --- 

        self.xbeach_params["nx"] = np.shape(zgr)[1] -1
        self.xbeach_params["ny"] = np.shape(zgr)[0] -1

        if self.drawfigs:
            # fig, ax = plt.subplots(1,1, figsize=(8,1))
            fig, ax = plt.subplots(1,1, figsize=(5,8))
            ax.pcolor(xgr,ygr,zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r")
            # ax.imshow(zgr, origin="lower", vmin=-8.5, vmax=8.5, cmap="BrBG_r")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)", rotation=90)
            ax.set_title("xbtg")
            if self.savefig:
                fn = os.path.join(self.path_to_figs, "xbeach-grid-interpolated.png")
                plt.savefig(fn, 
                            transparent=False, 
                            dpi=500,
                            bbox_inches="tight",
                            pad_inches=0.1,
                            )

            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.plot(xgr[zgrid_row,:],zgr[zgrid_row,:], "k")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_title("Transect at row: {}" .format(zgrid_row))
            if self.savefig:
                fn = os.path.join(self.path_to_figs, "transect{}.png" .format(zgrid_row))
                plt.savefig(fn, 
                            transparent=False, 
                            dpi=500,
                            bbox_inches="tight",
                            pad_inches=0.1,
                            )
        return xgr, ygr, zgr
    
    def add_buildings(self, xgr, ygr, zgr, grid_df, struct_height=10):
        """ function to add buildings to grid. 

        returns: 
            zgr: updated z-grid with values of struct_height where ever
                structures are located
            nesgr: a non-erodible structure grid that identifies which
                areas of the grid can be eroded. Values in this grid define 
                the thickness of the erodible layer on top of the non-erodible 
                layer. For example, 0 indicate no erosion is possible, whereas 
                values 10 indicate that up to 10m of erosion is possible. 
        """
        print("need to confirm this works with variable grid.")
        nesgr = np.ones_like(zgr)
        if self.path_to_buildings != None:
            gdf_buildings = gpd.read_file(self.path_to_buildings)
            gdf_buildings.to_crs(self.local_epsg, inplace=True)
        else:
            return zgr, nesgr
        grid_df = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.pt_x_wrld, grid_df.pt_y_wrld), crs=self.local_epsg)

        # loop through each buildling and finds points in building geom; change elevation to 99
        for i in range(len(gdf_buildings)):
            bldg_ = gdf_buildings.iloc[i]
            gdf_temp = bldg_.geometry.contains(grid_df.geometry)
            if gdf_temp.sum()>0:        # if there is a grid cell with a building on.
                grid_ = grid_df.loc[gdf_temp]
                zgr[grid_["idy"], grid_["idx"]] = struct_height

        if self.drawfigs:
            fig, ax = plt.subplots(1,1, figsize=(4,8))
            ax.pcolor(xgr,ygr,zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)", rotation=90)
            ax.set_title("grid-w-bldgs")
            if self.savefig:
                fn = os.path.join(self.path_to_figs, "grid-w-bldgs.png")
                plt.savefig(fn, 
                            transparent=False, 
                            dpi=500,
                            bbox_inches="tight",
                            pad_inches=0.1,
                            )

        # setting up non-erodible structure grid
        if self.xbeach_params["morphology"] == 1:
            if self.xbeach_params["struct"] == 1:
                nesgr = np.zeros(np.shape(zgr))
                nesgr[zgr!=struct_height] = struct_height
                if self.drawfigs:
                    fig, ax = plt.subplots(1,1, figsize=(4,8))
                    ax.pcolor(xgr, ygr, nesgr, vmin=0, vmax=struct_height)
                    ax.set_xlabel("x (m)")
                    ax.set_ylabel("y (m)")
                    ax.set_title("Non-erodible structure layer")
                    if self.savefig:
                        fn = os.path.join(self.path_to_figs, "non-erodible-struct-layer.png")
                        plt.savefig(fn, 
                                    transparent=False, 
                                    dpi=500,
                                    bbox_inches="tight",
                                    pad_inches=0.1,
                                    )

        return zgr, nesgr

    def setup_forcing(self):
        for file_i, file in enumerate(self.forcing_files):
            fn = os.path.join(self.path_to_forcing, file)
            df_ = self.frcing_to_dataframe(fn, file, t_start=40)
            
            if file_i == 0:
                frcng_df = df_.copy()
                frcng_df["el{}" .format(file_i+1)] = df_["el"]
                frcng_df["Hs{}" .format(file_i+1)] = df_["Hs"]
                frcng_df["Tp{}" .format(file_i+1)] = df_["Tp"]
                frcng_df["mainang{}" .format(file_i+1)] = df_["mainang"]
                del frcng_df["el"]
                del frcng_df["Hs"]
                del frcng_df["Tp"]
                del frcng_df["mainang"]
            else:
                frcng_df["el{}" .format(file_i+1)] = df_["el"]
                frcng_df["Hs{}" .format(file_i+1)] = df_["Hs"]
                frcng_df["Tp{}" .format(file_i+1)] = df_["Tp"]
                frcng_df["mainang{}" .format(file_i+1)] = df_["mainang"]

            

        # -- water elevations --
        """
            Note that Don's data is provided counterclockwise around domain, 
            whereas xbeach goes clockwise. Need to be careful with this. 
           4|---------|3
            |         |
            |  SWAN   |
            |         |
            |         |
           1|---------|2 

           2|---------|3
            |         |
            | XBEACH  |
            |         |
            |         |
           1|---------|4 

        """

        if self.xbeach_params["tideloc"] == 1:
            elev_df = frcng_df[["t_sec", "el1"]]
        else:
            elev_df = frcng_df[["t_sec", "el1", "el4", "el3", "el2"]]       # forcing points must be clockwise around domain starting from lower left corner

        fn_out = os.path.join(self.path_to_model, self.xbeach_params["zs0file"])
        elev_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')

        # -- wave forcing --
        if self.xbeach_params["nspectrumloc"] == 4:  # if more than one wave spectra provided
            adcirc_locs = [1, 4, 2, 2]                    # adcirc/swan locations for wave forcing. 
            for al in adcirc_locs:
                Hs_key = "Hs{}" .format(al)         # get Hs, Tp, and mainang for savepoint
                Tp_key = "Tp{}" .format(al)
                mainang_key = "mainang{}" .format(al)
                wave_df = frcng_df[[Hs_key, Tp_key, mainang_key, "gammajsp", "s", "duration", "dtbc"]]

                # write to jonswap file
                fn_out = os.path.join(self.path_to_model, "jonswap{}.txt" .format(al))
                wave_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')

            fn = os.path.join(self.path_to_model, "loclist.txt")
            with open(fn, 'w') as f:
                f.write("LOCLIST\n" .format(self.model_name))
                f.write("0. 0. jonswap{}.txt\n" .format(adcirc_locs[0]))
                f.write("0. {}. jonswap{}.txt\n" .format(self.xbeach_params["ny"], adcirc_locs[1]))
                f.write("{}. {}. jonswap{}.txt\n" .format(self.xbeach_params["nx"], self.xbeach_params["ny"], adcirc_locs[2]))
                f.write("{}. 0. jonswap{}.txt\n" .format(self.xbeach_params["nx"], adcirc_locs[3]))
        
        elif self.xbeach_params["nspectrumloc"] == 2:
            adcirc_locs = [1, 4]                    # adcirc/swan locations for wave forcing. 
            for al in adcirc_locs:
                Hs_key = "Hs{}" .format(al)         # get Hs, Tp, and mainang for savepoint
                Tp_key = "Tp{}" .format(al)
                mainang_key = "mainang{}" .format(al)
                wave_df = frcng_df[[Hs_key, Tp_key, mainang_key, "gammajsp", "s", "duration", "dtbc"]]

                # write to jonswap file
                fn_out = os.path.join(self.path_to_model, "jonswap{}.txt" .format(al))
                wave_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')

            fn = os.path.join(self.path_to_model, "loclist.txt")
            with open(fn, 'w') as f:
                f.write("LOCLIST\n" .format(self.model_name))
                f.write("0. 0. jonswap{}.txt\n" .format(adcirc_locs[0]))
                f.write("0. {}. jonswap{}.txt\n" .format(self.xbeach_params["ny"], adcirc_locs[1]))
        else:
            adcirc_loc = 4
            wave_df = frcng_df[["Hs{}" .format(adcirc_loc), "Tp{}" .format(adcirc_loc), "mainang{}" .format(adcirc_loc), "gammajsp", "s", "duration", "dtbc"]]

            fn_out = os.path.join(self.path_to_model, self.xbeach_params["bcfile"])
            wave_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')

        return frcng_df

    def frcing_to_dataframe(self, fn, filename, n_header=3, n_var=7, t_start=0):
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
                
                """ Getting correct wave angle.
                    - SWAN wave angles measured in cartesian 
                      - Cartesian convention: waves traveling TO the east are 0 
                        and counterclockwise is positive.
                    - XBEACH jonswap.txt file takes keyword `mainang` which is 
                      measured in nautical convention.
                      - Nautical convention: waves traveling FROM North are 0 and 
                        clockwise is positive. 
                      - Default XBEACH jonswap angle is 270 indicating that 
                        waves approach the shore from the west (e.g., travel 
                        west to east).
                    
                    - In this model setup, xbeach does not know angle that grid 
                      is rotated (e.g., I'm not providing alfa from the xbeach). 
                    - As such, every wave angle needs to be rotated by positive 
                      alfa
                """
                wavedir_ = self.cartesian_to_nautical_angle(wavedir_)
                wavedir_ = self.nautical_to_xbeach_angle(wavedir_, self.alfa)
                # wavedir_ = self.adjust_towards_shore(wavedir_)

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
        df["Hs"] = hs
        df["Tp"] = Tp
        # df["wavedir"] = wavedir

        df["el"] = df["el"]*0.3048
        df["Hs"] = df["Hs"]*0.3048
        # df["el_land"] = 0

        dt = (df["t"].iloc[1] - df["t"].iloc[0])*60*60         # tiime setp in seconds; converting from hours.
        df["t_sec"] = np.linspace(0, (len(df)-1)*dt, len(df))
        
        # ---
        # for testing
        # wavedir = 270
        # print("drs temporarily setting all angles to {}" .format(wavedir))
        """model returns NaN when using provided wave angles. 
           I think waves start offshore
        """
        # ---

        print("need to figureout s in jonswap params file")
        df["mainang"] = wavedir   # main wave angle
        df["gammajsp"] = 3.3    # peak enhancement factor for jonswap; not used in stationary mode
        df["s"] = 50          # directional spreading coeff (larger value results in longer wave crests)
        df["duration"] = dt   # duration of wave conditions
        df["dtbc"] = 0.5        # Timestep used to describe time series of wave energy and long wave flux at offshore boundary

        if t_start!=0:
            print("Starting at t={} hr of Don's simulation." .format(t_start))
            df = df.loc[df["t"]>=t_start]
            del df["t"]
            df["t"] = np.linspace(0, (len(df)-1)*dt/3600, len(df))
            df["t_sec"] = np.linspace(0, (len(df)-1)*dt, len(df))


        if self.drawfigs:
            fig, ax = plt.subplots(1,1)
            ax.plot(df["t_sec"], df["el"])
            ax.set_title(filename)
            if self.savefig:
                fn = os.path.join(self.path_to_figs, "{}.png" .format(filename))
                plt.savefig(fn, 
                            transparent=False, 
                            dpi=500,
                            bbox_inches="tight",
                            pad_inches=0.1,
                            )

        return df
    
    def cartesian_to_nautical_angle(self, deg):
        """ converting from cartesian to nautical angles for xbeach input
        Cartesian: waves traveling TO east are zero and counterclockwise is positive.
        Nautical: waves traveling FROM North are zero and clockwise is positive. 
        """
        if (deg>=0) & (deg <= 270):
           return (270-deg)
        elif (deg>270) & (deg<360):
           return (270-deg)+360
        else:
            raise ValueError("{} must be between 0 and 360." .format(deg))

    def nautical_to_xbeach_angle(self, deg, alfa):
        """
        """
        deg = deg + alfa 
        if deg > 360:
            deg -= 360
        elif deg < 0:
            deg += 360
        return deg

    def adjust_towards_shore(self, deg):
        print("Warning: manually adjusting wave angles to head towards shore.")
        if (deg < 180) & (deg>=90):
            deg = 180
        elif (deg>0) & (deg<90):
            deg = 0
        return deg


    def create_model(self, xgr, ygr, zgr, nesgr, elev_df):
        # writing model using xbeachtools
        xb_setup = XBeachModelSetup(self.model_name)
        xb_setup.set_grid(xgr, ygr, zgr, posdwn=-1) # alfa=self.xbeach_params["alfa"])
        xb_setup.set_nebed(nesgr)
        xb_setup.set_friction(np.ones_like(zgr))
        xb_setup.set_params({})
        xb_setup.write_model(self.path_to_model)

        self.rename_file("bed.dep", self.xbeach_params["depfile"])
        self.rename_file("x.grd", self.xbeach_params["xfile"])
        self.rename_file("y.grd", self.xbeach_params["yfile"])
        if self.xbeach_params["struct"]==1:
            self.rename_file("ne_bed.dep", self.xbeach_params["ne_layer"])

        # now writing own params file
        os.remove(os.path.join(self.path_to_model, "params.txt"))
        os.remove(os.path.join(self.path_to_model, "friction.dep"))
        if self.xbeach_params["struct"]==0:
            os.remove(os.path.join(self.path_to_model, "ne_bed.dep"))
        
        if self.xbeach_params["tstop"] == None:
            self.xbeach_params["tstop"] = elev_df.iloc[-1]["t_sec"].astype(int)
        
        if self.xbeach_params["vardx"] == 0:
            # -- setting dx/dy from xbeach_res
            self.xbeach_params["dx"] = self.xbeach_params["xbeach_res"]
            self.xbeach_params["dy"] = self.xbeach_params["xbeach_res"]
            
            # -- deleting xfile and yfile
            # os.remove(os.path.join(self.path_to_model, self.xbeach_params["xfile"]))
            # os.remove(os.path.join(self.path_to_model, self.xbeach_params["yfile"]))
            
            # -- deleting xfile and yfile keys
            del self.xbeach_params["xfile"]
            del self.xbeach_params["yfile"]

        self.write_xbeach_params()
        self.move_figs()

    def rename_file(self, fn_old, fn_new):
        fn_old = os.path.join(self.path_to_model, fn_old)
        fn_new = os.path.join(self.path_to_model, fn_new)
        os.rename(fn_old, fn_new)


    def write_xbeach_params(self):
        grid_input_keys = ["nx", "ny", "dx", "dy", "xori", "yori", "alfa", "depfile", "vardx", "xfile", "yfile", "posdwn", "thetamin", "thetamax", "dtheta",  "dtheta_s", "wavint"]
        numerics_input_keys = ["CFL", "eps", "front", "back", "scheme", "left", "right", "maxdtfac"]
        time_input_keys = ["dt", "tstart", "tintg", "tintm", "tintp", "tstop", "taper", "dtset"]
        general_constants = ["rho", "g"]
        boundary_condition_keys = ["zs0file", "tideloc", "tidetype", "tidelen", "zs0", "bcfile", "rt", "dtbc", "sprdthr", "wbcversion", "nspectrumloc"]
        wave_calculation_keys = ["wavemodel", "wbctype", "instat", "break", "wci", "roller", "beta", "gamma", "gammax", "alpha", "delta", "n", "maxerror", "maxiter"]
        flow_calculation_keys = ["nuh", "nuhfac", "nuhv", "umin"]
        sed_trans_calculation_keys = ["sedtrans", "dico", "D50", "D90", "rhos", "z0"]
        morphological_calculation_keys = ["morphology", "struct", "ne_layer", "morfac", "morstart", "por", "dryslp", "wetslp", "hswitch"]

        fn = os.path.join(self.path_to_model, "params.txt")
        with open(fn, 'w') as f:
            f.write("XBeach Model : {}\n" .format(self.model_name))
            f.write("Params Written : {}\n" .format(datetime.now().replace(second=0, microsecond=0)))

            self.write_xbeach_params_section(f, "Grid Input (res python: {})" .format(self.xbeach_params["xbeach_res"]), grid_input_keys)
            self.write_xbeach_params_section(f, "Numerics Input",       numerics_input_keys)
            self.write_xbeach_params_section(f, "Time Input",           time_input_keys)
            self.write_xbeach_params_section(f, "General Constants",    general_constants)
            self.write_xbeach_params_section(f, "Boundary Conditions",  boundary_condition_keys)
            self.write_xbeach_params_section(f, "Wave Calculation Options",     wave_calculation_keys)
            self.write_xbeach_params_section(f, "Flow Calculation Options",     flow_calculation_keys)
            self.write_xbeach_params_section(f, "Sediment Transport Options",   sed_trans_calculation_keys)
            self.write_xbeach_params_section(f, "Morphological Calculation Options", morphological_calculation_keys)

            f.write("\n-----------------------------------------\n" )
            f.write("Output Options\n")
            f.write("{:20s} = {}\n" .format("nglobalvar", len(self.xbeach_params["global_var"])))
            for i in self.xbeach_params["global_var"]:
                f.write("{}\n" .format(i))

    def write_xbeach_params_section(self, f, title, key_list):
            f.write("\n-----------------------------------------\n" )
            f.write("\n{}\n" .format(title))
            for key in key_list:
                if key in self.xbeach_params.keys():
                    f.write("{:20s} = {}\n" .format(key, str(self.xbeach_params[key])))



    def move_figs(self):
        files_in_dir = os.listdir(self.path_to_model)
        for file in files_in_dir:
            if ".png" in file:
                os.replace(os.path.join(self.path_to_model,file), os.path.join(self.path_to_figs,file))

    def reproject_raster(self):
        """
        function to reproject raster from current epsg to local utm epsg.
        Note that a temporary tiff file is written to "temp.tiff". This file is
        later removed when it is no longer needed.
        """
        with rasterio.open(self.path_to_dem) as src:
            transform, width, height = calculate_default_transform(
                src.crs, self.local_epsg, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": self.local_epsg,
                "transform": transform,
                "width": width,
                "height": height
            })

            with rasterio.open("temp.tiff", "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    d = reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.local_epsg,
                        resampling=Resampling.nearest)

    def setup_grid(self, gdf_domain):
        """ 
        function to setup empty xbeach grid. 
        returns an empty numpy array with dimensions equal to the size of 
        the input grid divided by the input resolution.

        returns:
            grid_gdf: geodataframe of the input grid.
            grid: elevation saved to a numpy array representing the grid
        """
        # getting exterior points 
        x = gdf_domain.geometry.exterior[0].xy[0]
        y = gdf_domain.geometry.exterior[0].xy[1]

        """ loop through exterior points; pull out grid width, length, origin, 
            and angle of rotation (theta).
        """
        l, w = [], []           # used to store both lengths and widths
        xo, yo = np.inf, np.inf         # x/y used for origin
        xa, ya = 0, 0                   # x/y used for calculating angle
        sides = []
        for i in range(len(x)):
            if i == 4:
                break
            # getting two points to calculate length/width
            x0, x1 = x[i], x[i+1]
            y0, y1 = y[i], y[i+1]
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            d = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
            sides.append(d)     # appending side length

            if y0<yo:       # getting origin point; crudely done. 
                xo, yo = x0, y0
            
            if x0>xa:       # getting point for calculating angle; crudely done
                xa, ya = x0, y0
        
        sides = np.array(sides)
        
        # note: the two lines below are used for small domain.
        # w = sides[np.argsort(sides)][2:]        # width  is defined here as crossshore distance
        # l = sides[np.argsort(sides)][0:2]       # length is defined here as alongshore distance
        
        # note: the two lines below are used for large domain.
        w = sides[np.argsort(sides)[0:2]]       # width  is defined here as crossshore distance
        l = sides[np.argsort(sides)[2:]]        # length is defined here as alongshore distance
        
        theta_r, theta_d = self.compute_theta((xo, yo), (xa, ya))   # angle that grid is rotated, measured relative to east
        self.origin = (xo, yo)
        self.alfa = theta_d         # storing angle of grid rotation to self.alfa
        # self.xbeach_params["alfa"] = theta_d         # rotation angle in model 

        l = self.myround(np.average(l), base=self.xbeach_params["xbeach_res"])
        w = self.myround(np.average(w), base=self.xbeach_params["xbeach_res"])

        nx = int(w/self.xbeach_params["xbeach_res"])
        ny = int(l/self.xbeach_params["xbeach_res"])

        grid = np.zeros((nx,ny))
        pt_x, pt_y, pt_x_wrld, pt_y_wrld, elev, idx, idy = [], [], [], [], [], [], []
        with rasterio.open("temp.tiff", "r") as src:
            for x in range(nx):
                for y in range(ny):
                    pt_x.append(x*self.xbeach_params["xbeach_res"])
                    pt_y.append(y*self.xbeach_params["xbeach_res"])

                    pt_x_wrld_ = xo + pt_x[-1]*np.cos(theta_r) - pt_y[-1]*np.sin(theta_r)
                    pt_y_wrld_ = yo + pt_x[-1]*np.sin(theta_r) + pt_y[-1]*np.cos(theta_r)
                    pt_x_wrld.append(pt_x_wrld_)
                    pt_y_wrld.append(pt_y_wrld_)
                    idx.append(x)
                    idy.append(y)

                    z = src.sample([(pt_x_wrld_, pt_y_wrld_)])
                    z = next(z)[0]
                    grid[x,y] = z
                    elev.append(z)

        # formatting data to pass back
        xvals = np.linspace(0, (nx-1)*self.xbeach_params["xbeach_res"], nx)
        yvals = np.linspace(0, (ny-1)*self.xbeach_params["xbeach_res"], ny)

        grid_df = pd.DataFrame()
        grid_df["pt_x"] = pt_x
        grid_df["pt_y"] = pt_y
        grid_df["pt_x_wrld"] = pt_x_wrld
        grid_df["pt_y_wrld"] = pt_y_wrld
        grid_df["elev"] = elev
        grid_df["idx"] = idx
        grid_df["idy"] = idy
        return grid_df, grid, xvals, yvals
    
    def compute_theta(self, pt1, pt2):
        """
        compute angle between two points.
        returns angle in both radians and degrees
        """
        h = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
        o = pt2[1] - pt1[1]
        theta_r = np.asin(o/h)
        theta_d = np.rad2deg(theta_r)
        return theta_r, theta_d


    def myround(self, x, base=5):
        return base * round(x/base)

    def make_directory(self, path_out):
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        return path_out


if __name__ == "__main__":
    sxb = setup_xbeach()
    sxb.setup_model()

    plt.show()



