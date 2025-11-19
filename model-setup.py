import os, shutil
import sys
import math
import numpy as np
from datetime import datetime
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns

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
        "model_name": "run65b",
        "duration": 4,          # time step (hr) in adcirc/swan time to start running XBeach
        "elevated_bldgs": True,
        "spatial_join_elevated": False,     # if doing spatial join to merge footprints with FFE, else use FolioID to merge

        "path_to_ffe": os.path.join(self.file_dir, "..",  "data", "mehrshad", "data", "Geoscience-collection--overall-dataset", "data", "FMB_VDA_FFE_Final.csv"),
        "path_to_dem": os.path.join(self.file_dir, "..",  "data", "dem", "dem-resampled.tiff"),

        "path_to_domain": os.path.join(self.file_dir, "..", "data", "xbeach-domain", "xbeach-domain-epsg32617.geojson"),
        # "path_to_domain": os.path.join(self.file_dir, "..",  "data", "xbeach-domain", "xbeach-domain-micro-epsg32617.geojson"),

        # "path_to_buildings": None,
        "path_to_buildings": os.path.join(self.file_dir, "..", "data", "buildings", "amini-bldgs-estero.geojson"),

        "path_to_forcing": os.path.join(self.file_dir, "..", "data", "forcing", "2025-10-24-update"),
        "forcing_ids": ["ff71b097", "fbcc9a44"],     # forcing point IDs starting from lower left corner or going clockwise (XBeach notation)
        # "forcing_ids": ["f4fad26a", "0199419d"],     # forcing point IDs starting from lower left corner or going clockwise (XBeach notation)

        "smooth_grid": True,
        "extend_grid": True,
        "save_pts_geojson": None,
        "local_utm_epsg": "EPSG:32617",
        "drawfigs": True,
        "savefigs": True,
        }
        return inputs

    def set_xbeach_params(self):
        self.xbeach_params = {
                    "xbeach_res" : 2,          # in local utm units (m here).
                    
                    # -- grid input --
                    # "xori"      : 0,            # x-coordinate of origin of axis
                    # "yori"      : 0,            # y-coordinate of origin of axis
                    "depfile"   : "z.grd",      # Name of the input bathymetry file
                    "vardx"     : 0,            # Switch for variable grid spacing
                    "xfile"     : "x.grd",      # Name of the file containing x-coordinates of the calculation grid
                    "yfile"     : "y.grd",      # Name of the file containing y-coordinates of the calculation grid
                    "posdwn"    : -1,           # Bathymetry is specified positive down (1) or positive up (-1)
                    # "thetamax"  : 90,           # Higher directional limit (angle w.r.t computational x-axis)   
                    # "thetamin"  : -90,          # Lower directional limit (angle w.r.t computational x-axis)
                    # # "single_dir": 0,            # Turn on stationary model for refraction, surfbeat based on mean direction
                    # "dtheta"    : 10,           # Directional resolution; 
                    # "dtheta_s"  : 10,           # Directional in case of stationary refraction; not used in stationary mode

                    # -- numerics input --
                    # "CFL"       : 0.5,          # Maximum courant-friedrichs-lewy number
                    # "eps"       : 0.001,        # Threshold water depth above which cells are considered wet
                    # "front"     : "wlevel",     # Switch for seaward flow boundary (abs_1d, abs_2d, wall, wlevel, nonh_1d, waveflume); switches to abs_1d for
                    # "back"      : "wlevel",     # Switch for boundary at bay side (wall, abs_1d, abs_2d, wlevel)   
                    # "scheme"    : "warmbeam",   # Numerical scheme for wave propagation (upwind_1, lax_wendroff, upwind_2, warmbeam)
                    "cyclic"    : 1,    # Turn on cyclic boundary conditions (0 or 1)
                    # "left"      : "neumann",    # Switch for lateral boundary at ny+1 (neumann, wall, no_advec, neumann_v, abs_1d)
                    # "right"     : "neumann",    # Switch for lateral boundary at 0    (neumann, wall, no_advec, neumann_v, abs_1d)
                    # "maxdtfac"     : 500,    # Maximum increase/decrease in time stp in explosion prevention mechanism
                    # "mmpi"        : 3,            # Number of domains in cross-shore direction when manually specifying mpi domains
                    # "nmpi"        : 3,            # Number of domains in alongshore direction when manually specifying mpi domains
                    # "mpiboundary" : "man",        # Fix mpi boundaries along y-lines, x-lines, use manual defined domains or find shortest boundary automatically
                    "outputprecision": "single",    # Switch between single and double precision output in netcdf (default: double)
                    "random"    : 0,            # Switch to enable random seed for instat = jons, swan or vardens boundary conditions

                    # -- time input --
                    "tstart"    : 0,            # Start time of output, in morphological time
                    "tintg"     : 0.5,         # interval time of global output
                    "tintm"     : 400,          # interval time of mean, var, max, min output
                    "tintp"     : 0.5,          # interval time of point/runup gauge output
                    # "tstop"     : 200000,     # end time seconds
                    "taper"     : 200,          # Spin-up time of wave boundary conditions, in morphological time
                    # "dtset"     : 0.1,        # Fixed timestep, overrides use of cfl

                    # -- general constants --
                    # "rho"   : 1025,             # Density of water
                    # "g"     : 9.81,             # Gravitational acceleration

                    # -- boundary conditions --
                    "zs0file"   : "water_elev.dat", # Name of tide boundary condition series
                    "tideloc"   : 1,                # Number of corner points on which a tide time series is specified
                    # "tidetype"  : "hybrid",         # Switch for offfshore boundary, velocity boundary or instant water level boundary (instant, velocity, hybrid; default velocity)
                    # "zs0"       : 0,              # Inital water level
                    # "paulrevere": "sea" ,         # Specifies tide on sea and land or two sea points if tideloc = 2 (land, sea)
                    # "tidelen"   : None,           # length of tide signal (doesn't appear to be read in xbeach)
                    "wind"      : 1,
                    "windfile"  : "wind.txt",        # name of windfile

                    # -- swan wave input options
                    "wbctype"   : "swan",           # swan wave input
                    "bcfile"    : "loclist.txt",    # Name of spectrum file; use if providing multiple spectra (nspectrumloc>1)
                    "nspectrumloc": 2,              # number of wave spectra in offshore boundary
                    # "dthetaS_XB": 0,              # [Note: variable set later in code] The (counter-clockwise) angle in the degrees needed to rotate from the x-axis in swan to the x-axis pointing east

                    # -- wave calculation options
                    # "wavemodel" : "nonh",     # (0) stationary, (1) surfbeat or (2) nonh
                    # "wbctype"   : "jonstable",  # New wave boundary condition type
                    # "nspectrumloc": 1,            # number of wave spectra in offshore boundary
                    # "bcfile"    : "jonswap.txt",         # Name of spectrum file
                    # "bcfile"    : "loclist.txt",         # Name of spectrum file; use if providing multiple spectra (nspectrumloc>1)

                    # "wbcversion"  : 3,            # wave boundary condition version
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
                    "global_var": ["zs1"], # "H",  "zs1", "hh", "u", "v", "ue", "ve"]
                    # "global_var": ["H", "zs"]   # when running surfbeat, need to collect "H".
                    # "point_var": ["zs", "zs0", "zs1", "H", "hh", "zb"]
                    }
    
    def setup_inputs(self):
        input_vals = self.input_vals()
        self.set_domain_micro(input_vals["path_to_domain"])
        self.set_model_name(input_vals["model_name"])
        self.set_t_start_stop(input_vals["duration"])
        self.set_local_utm_epsg(input_vals["local_utm_epsg"])

        self.set_elevated_bldgs(input_vals["elevated_bldgs"])
        self.set_spatial_join(input_vals["spatial_join_elevated"])
        self.set_path_to_ffe(input_vals["path_to_ffe"])

        self.set_path_to_dem(input_vals["path_to_dem"])
        self.set_path_to_domain(input_vals["path_to_domain"])
        self.set_path_to_buildings(input_vals["path_to_buildings"])
        self.set_smooth_grid(input_vals["smooth_grid"])
        self.set_extend_grid(input_vals["extend_grid"])

        self.set_forcing_inputs(input_vals["path_to_forcing"], input_vals["forcing_ids"])
        self.set_save_pts_geojson(input_vals["save_pts_geojson"])
        self.set_drawfigs(input_vals["drawfigs"])
        self.set_savefigs(input_vals["savefigs"])

    def set_domain_micro(self, val):
        if "micro" in val:
            self.microdomain = True
        else:
            self.microdomain = False

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.path_to_model = os.path.join(self.file_dir, "models", model_name)
        self.make_directory(self.path_to_model)

    def set_t_start_stop(self, val):
        duration_to_start_stop = {
                    0.5: {"start": 66.25, "stop":  66.75},
                    1:   {"start": 66,    "stop":  67},
                    2:   {"start": 65.25, "stop":  67.25},
                    3:   {"start": 65,    "stop":  68},
                    4:   {"start": 64,    "stop":  68},
                    6:   {"start": 63,    "stop":  69},
                    8:   {"start": 62,    "stop":  70},
                    12:  {"start": 60,    "stop":  72}
                                }
        self.t_start = duration_to_start_stop[val]["start"]
        self.t_stop  = duration_to_start_stop[val]["stop"]

    def set_elevated_bldgs(self, val=None):
        self.elevated_bldgs = val

    def set_spatial_join(self, val=None):
        self.spatial_join = val

    def set_path_to_ffe(self, val=None):
        self.path_to_ffe = val

    def set_path_to_dem(self, val=None):
        self.path_to_dem = val

    def set_path_to_domain(self, val=None):
        self.path_to_domain = val

    def set_path_to_buildings(self, val=None):
        self.path_to_buildings = val
    
    def set_smooth_grid(self, val):
        self.smooth_grid_tf = val
    
    def set_extend_grid(self, val):
        self.extend_grid_tf = val

    def set_forcing_inputs(self, val, frcng_ids):
        self.path_to_forcing = val
        self.frcng_ids = frcng_ids
        
        all_forcing_files = os.listdir(self.path_to_forcing)
        self.water_elevation_files = [i for i in all_forcing_files if ".dat" in i]
        self.swan_file = [i for i in all_forcing_files if ".out" in i]

        fn_pts_geojson = [i for i in all_forcing_files if ".geojson" in i][0]
        self.frcng_pts_gdf = gpd.read_file(os.path.join(self.path_to_forcing, fn_pts_geojson))
        self.frcng_pts_gdf.set_index("uid", inplace=True)
        self.frcng_pts_gdf.to_crs(self.local_epsg)
        self.frcng_pts_gdf["swan_order"] = self.frcng_pts_gdf["swan_order"].astype("Int64")
    
    def set_save_pts_geojson(self, val):
        self.save_pt_geojson = val

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
        grid_df = self.raster_to_xbeach_grid()             # from raster to rotated xbeach grid.
        # xgr, ygr, zgr = self.xbtools_grid(bathy, x, y)                  # using xbeach tools to prepare xbeach grid
        grid_df, nesgr = self.add_buildings(grid_df)
        savepoint_df = self.setup_savepoints(grid_df)
        self.setup_forcing(grid_df)
        self.create_model(grid_df, nesgr, savepoint_df) # writing out xbeach model

    def raster_to_xbeach_grid(self):
        """
        reads raster data, converts to local crs (local crs units must be m.),
        crops raster to xbeach-domain, setsup an empty rotated grid, then 
        cell-by-cell populates the empty grid with the dem elevations.
        Results in an xbeach grid that is orientied with (0,0) in the lower 
        left-hand corner. 
        """
        print("Preparing grid: ")
        gdf_domain = gpd.read_file(self.path_to_domain)     # read in xbeach domain
        gdf_domain.to_crs(self.local_epsg, inplace=True)

        self.reproject_raster()                             # reproject raster to local crs
        grid_df, grid, x, y = self.setup_grid(gdf_domain)   # setup xbeach grid with (0,0) in lower left corner. grid is rotated.
        os.remove("temp.tiff")
        
        if self.drawfigs:
            # -- plotting grid; no buildings
            if self.microdomain:
                figsize=(5,4)
            else:
                figsize=(3,8)
            fig, ax = plt.subplots(1,1, figsize=figsize)
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

            # -- plotting one transect in x direction; halfway up domain
            zgrid_row=None
            if zgrid_row==None:
                zgrid_row = int(np.rint(np.shape(grid)[1]/2))
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.plot(x, grid[:,zgrid_row], "k")
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

            # -- plotting a few transects in y-direction
            xgr, ygr, zgr = self.grid_df_to_xyz(grid_df)
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            x_trans = [0, 50, 100, 200, 300]
            colors = sns.color_palette("viridis", len(x_trans))
            cnt = 0
            for x_trans_ in x_trans:
                idx = np.argmin(np.abs(xgr[0,:] - x_trans_))

                y_data = ygr[:,idx]
                # y_data = y_data[::-1]
                z_data = zgr[:,idx]

                ax.plot(y_data, z_data, color=colors[cnt], label="x={}" .format(x_trans_))
                cnt += 1
            ax.xaxis.set_inverted(True)  # inverted axis with autoscaling
            ax.grid()
            ax.set_title("Elevation\nFrom Sea, Looking Towards Land")
            # ax.set_ylim([-3.5, 1.])
            ax.legend(loc="upper right", ncols=len(x_trans))

            if self.savefig:
                fn = os.path.join(self.path_to_figs, "y-transects.png")
                plt.savefig(fn, 
                            transparent=False, 
                            dpi=500,
                            bbox_inches="tight",
                            pad_inches=0.1,
                            )


        print("  Done.")
        return grid_df
    
    def grid_df_to_xyz(self, grid_df):
        xvals = grid_df["pt_x"].unique()
        yvals = grid_df["pt_y"].unique()
        xgrd, ygrd = np.meshgrid(xvals, yvals)

        zgrd = np.zeros((len(xvals),len(yvals)))
        zgrd[grid_df["idx"], grid_df["idy"]] = grid_df["elev"]

        return xgrd, ygrd, zgrd.T       

    def write_grid(self, grid_df):
        xgrd, ygrd, zgrd = self.grid_df_to_xyz(grid_df)

        fn_outx = os.path.join(self.path_to_model, "x.grd")
        fn_outy = os.path.join(self.path_to_model, "y.grd")
        fn_outz = os.path.join(self.path_to_model, "z.grd")

        np.savetxt(fn_outx, xgrd, fmt='%.3f', delimiter=' ')
        np.savetxt(fn_outy, ygrd, fmt='%.3f', delimiter=' ')
        np.savetxt(fn_outz, zgrd, fmt='%.3f', delimiter=' ')

    
    def smooth_grid(self, z, D=1.0, dt=0.1, num_steps=10):
        """
        Smooths a 2D field z(x,y) using a 2nd order diffusion operator.
        
        Args:
            z (np.ndarray): The 2D field to be smoothed, with shape (Nx, Ny).
            D (float): The diffusion coefficient (default=1)
            dt (float): The time step for the forward Euler scheme (default=0.1)
            num_steps (int): The number of time steps to run the smoothing (default=20).
            
        Returns:
            np.ndarray: The smoothed 2D field.
        """
        Nx, Ny = z.shape
        Lx = self.xbeach_params["xbeach_res"]*Nx  # The domain size in the x-direction.
        Ly = self.xbeach_params["xbeach_res"]*Ny  # The domain size in the y-direction.

        dx = Lx / (Nx - 1)
        dy = Ly / Ny
        
        # Ensure stability: D*dt/dx^2 and D*dt/dy^2 should be <= 0.25 (for forward Euler)
        if D * dt / dx**2 > 0.25 or D * dt / dy**2 > 0.25:
            print("Warning: The time step `dt` may be too large for stability.")
            print("Recommended max dt:", 0.25 * min(dx**2, dy**2) / D)
        
        # Create a copy to avoid modifying the original array
        z_smoothed = np.copy(z)
        
        for _ in range(num_steps):
            # Calculate the Laplacian (diffusion operator)
            laplacian = np.zeros_like(z_smoothed)
            
            # Diffusion in x-direction (solid boundaries)
            laplacian[1:-1, :] += (z_smoothed[2:, :] - 2 * z_smoothed[1:-1, :] + z_smoothed[:-2, :]) / dx**2
            
            # Diffusion in y-direction (periodic boundaries)
            laplacian[:, 1:-1] += (z_smoothed[:, 2:] - 2 * z_smoothed[:, 1:-1] + z_smoothed[:, :-2]) / dy**2
            
            # Handle periodic y-boundaries explicitly for the first and last columns
            # laplacian[:, 0]  += (z_smoothed[:, 1] - 2 * z_smoothed[:, 0]  + z_smoothed[:, -1]) / dy**2
            # laplacian[:, -1] += (z_smoothed[:, 1] - 2 * z_smoothed[:, -1] + z_smoothed[:, -2]) / dy**2
            
            # Update the field using the forward Euler method
            z_smoothed += D * dt * laplacian
            
        return z_smoothed

    def expand_grid_top(self, grid, grid_df, xo, yo, theta_r, pad=10):
        """
        function to expand the grid in the y-direction at the top of the domain.
        used when cyclic BCs are turned on.

        pad: length to pad in y-direction;
        """
        y = grid_df["pt_y"].unique()
        y_end = y[-1]
        dy = (y[1]-y[0])
        y_new = np.arange(y_end+dy, (y_end+pad)+dy, dy)
        y_out = np.append(y, y_new)

        # -- getting start and end values for z and y
        y1 = y_new[0]
        z1 = grid[:,0]
        y2 = y_new[-1]
        z2 = grid[:,-1]

        # -- sin^2 smoothing
        t = (y_new - y1) / (y2 - y1)
        theta = np.pi/2 + (np.pi/2) * t
        s = np.sin(theta)**2
        s = np.stack((s,) * len(z1), axis=0)

        # -- appending to grid
        z_new = z1[:,None] + ((z2-z1)[:, None] * s)
        grid_new = np.column_stack((grid,z_new))

        # -- setting up grid of indicies; flattening to put in grid_df
        idx = grid_df["idx"].unique()
        idy = np.arange(0, int(pad/dy)) + grid_df["idy"].max() + 1

        idx, idy = np.meshgrid(idx,idy)
        idx, idy = idx.flatten(), idy.flatten()

        # -- setting up pt_x and pt_y; this is with XBeach resolution
        pt_x = grid_df["pt_x"].unique()
        pt_y = y_new.copy()
        pt_x, pt_y = np.meshgrid(pt_x, pt_y)
        pt_x, pt_y = pt_x.flatten(), pt_y.flatten()

        # -- setting x and y coordinates in real-world points; uses origin above which is in model crs
        pt_x_wrld = xo + pt_x*np.cos(theta_r) - pt_y*np.sin(theta_r)
        pt_y_wrld = yo + pt_x*np.sin(theta_r) + pt_y*np.cos(theta_r)


        # -- loop through coordinates and get elevations
        # print("need to confirm this is necessary")
        # coord_list = [(x, y) for x, y in zip(pt_x_wrld, pt_y_wrld)]
        # with rasterio.open("temp.tiff", "r") as src:
        #     elev = [x[0].item() for x in src.sample(coord_list)]

        grid_df_new = pd.DataFrame()
        grid_df_new["pt_x"] = pt_x
        grid_df_new["pt_y"] = pt_y
        grid_df_new["pt_x_wrld"] = pt_x_wrld
        grid_df_new["pt_y_wrld"] = pt_y_wrld
        grid_df_new["idx"] = idx
        grid_df_new["idy"] = idy
        
        grid_df["elev"] = grid_new[grid_df["idx"], grid_df["idy"]]

        # grid_df_new["elev"] = elev

        grid_df = pd.concat([grid_df, grid_df_new], ignore_index=True)

        return grid_new, grid_df
    
    def repeat_grid_bottom(self, grid, grid_df, xo, yo, theta_r, pad=10):
        y = grid_df["pt_y"].unique()
        y_start = int(y[0])
        y_end = int(y[-1])

        dy = int(y[1]-y[0])

        y_out = np.array(range(y_start, y_end+pad, dy))

        # -- smooth and extend
        y_new = np.arange(0, pad, dy)
        y_new = np.flip(y_new)
        y1 = pad
        z1 = grid[:,0]
        y2 = 0
        z2 = np.average(grid[:,0])

        t = (y_new - y1) / (y2 - y1)
        theta = np.pi/2 + (np.pi/2) * t
        s = np.sin(theta)**2
        S = np.stack((s,) * len(z1), axis=0)
        z_new = z1[:,None] + ((z2-z1[:, None]) * s)
        grid_new = np.column_stack((z_new, grid))
        grid_new = self.smooth_grid(grid_new)

        # ---
        

        # -- setting up x and y as x-y points (0, 1, 2, 3)
        nx, ny = np.shape(grid)[0], len(y_out)
        x = np.arange(0, nx)
        y = np.arange(0, ny)
        
        # -- setting up grid of indicies; flattening to put in grid_df
        idx, idy = np.meshgrid(x, y)
        idx, idy = idx.flatten(), idy.flatten()

        # -- setting up pt_x and pt_y; this is with XBeach resolution
        pt_x = x*self.xbeach_params["xbeach_res"]
        pt_y = y*self.xbeach_params["xbeach_res"]

        # -- setting up grid of pt_x and pt_y; flattening to put in grid_df
        pt_x, pt_y = np.meshgrid(pt_x, pt_y)
        pt_x, pt_y = pt_x.flatten(), pt_y.flatten()

        # -- getting new (xo, yo) for grid expanded at the bottom
        h = pad   # hypotenuse
        xp = xo + h*np.sin(theta_r)
        yp = yo - h*np.cos(theta_r)

        # -- setting x and y coordinates in real-world points; uses origin above which is in model crs
        pt_x_wrld = xp + pt_x*np.cos(theta_r) - pt_y*np.sin(theta_r)
        pt_y_wrld = yp + pt_x*np.sin(theta_r) + pt_y*np.cos(theta_r)

        grid_df = pd.DataFrame()
        grid_df["pt_x"] = pt_x
        grid_df["pt_y"] = pt_y
        grid_df["pt_x_wrld"] = pt_x_wrld
        grid_df["pt_y_wrld"] = pt_y_wrld
        grid_df["idx"] = idx
        grid_df["idy"] = idy
        grid_df["elev"] = grid_new[grid_df["idx"], grid_df["idy"]]
        grid_new = np.zeros((nx, ny))
        grid_new[grid_df["idx"], grid_df["idy"]] = grid_df["elev"]

        return grid_new, grid_df


    def expand_grid_bottom(self, grid, grid_df, xo, yo, theta_r, pad=10):
        """
        function to expand the grid in the y-direction at the top of the domain.
        used when cyclic BCs are turned on.

        pad: number of cells to pad in y-direction;
        """
        y = grid_df["pt_y"].unique()
        y_start = int(y[0])
        y_end = int(y[-1])

        dy = int(y[1]-y[0])
        y_out = np.array(range(y_start, y_end+dy*pad, dy))

        # -- setting up x and y as x-y points (0, 1, 2, 3)
        nx, ny = np.shape(grid)[0], len(y_out)
        x = np.arange(0, nx)
        y = np.arange(0, ny)
        
        # -- setting up grid of indicies; flattening to put in grid_df
        idx, idy = np.meshgrid(x, y)
        idx, idy = idx.flatten(), idy.flatten()

        # -- setting up pt_x and pt_y; this is with XBeach resolution
        pt_x = x*self.xbeach_params["xbeach_res"]
        pt_y = y*self.xbeach_params["xbeach_res"]

        # -- setting up grid of pt_x and pt_y; flattening to put in grid_df
        pt_x, pt_y = np.meshgrid(pt_x, pt_y)
        pt_x, pt_y = pt_x.flatten(), pt_y.flatten()

        # -- getting new (xo, yo) for grid expanded at the bottom
        h = pad*dy   # hypotenuse
        xp = xo + h*np.sin(theta_r)
        yp = yo - h*np.cos(theta_r)

        # -- setting x and y coordinates in real-world points; uses origin above which is in model crs
        pt_x_wrld = xp + pt_x*np.cos(theta_r) - pt_y*np.sin(theta_r)
        pt_y_wrld = yp + pt_x*np.sin(theta_r) + pt_y*np.cos(theta_r)
        
        coord_list = [(x, y) for x, y in zip(pt_x_wrld, pt_y_wrld)]
        with rasterio.open("temp.tiff", "r") as src:
            elev = [x[0].item() for x in src.sample(coord_list)]

        grid_df = pd.DataFrame()
        grid_df["pt_x"] = pt_x
        grid_df["pt_y"] = pt_y
        grid_df["pt_x_wrld"] = pt_x_wrld
        grid_df["pt_y_wrld"] = pt_y_wrld
        grid_df["idx"] = idx
        grid_df["idy"] = idy
        grid_df["elev"] = elev
        # grid_df["elev"] = grid_new[grid_df["idx"], grid_df["idy"]]
        grid_new = np.zeros((nx, ny))
        grid_new[grid_df["idx"], grid_df["idy"]] = grid_df["elev"]


        return grid_new, grid_df

    def add_buildings(self, grid_df, struct_height=10):
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
        print("Adding buildings: ")
        xgr, ygr, zgr = self.grid_df_to_xyz(grid_df)
        zgr_original = np.copy(zgr)
        nesgr = np.ones_like(zgr)
        if self.path_to_buildings != None:
            gdf_buildings = gpd.read_file(self.path_to_buildings)
            gdf_buildings.to_crs(self.local_epsg, inplace=True)
            if self.elevated_bldgs:
                gdf_buildings = self.remove_elevated_bldgs(gdf_buildings, spatial_join=self.spatial_join)
        else:
            return grid_df, nesgr

        grid_df = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.pt_x_wrld, grid_df.pt_y_wrld), crs=self.local_epsg)

        # loop through each buildling and finds points in building geom; change elevation to 99
        for i in range(len(gdf_buildings)):
            bldg_ = gdf_buildings.iloc[i]
            gdf_temp = bldg_.geometry.contains(grid_df.geometry)
            if gdf_temp.sum()>0:        # if there is a grid cell with a building on.
                grid_ = grid_df.loc[gdf_temp]
                zgr[grid_["idy"], grid_["idx"]] = struct_height


        if self.drawfigs:
            if self.microdomain:
                figsize=(5,4)
            else:
                figsize=(3,8)
            fig, ax = plt.subplots(1,1, figsize=figsize)

            ax.pcolor(xgr,ygr,zgr, vmin=-8.5, vmax=8.5, cmap="BrBG_r")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)", rotation=90)
            ax.set_title("grid-w-bldgs")
            ax.set_aspect("equal")
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
                nesgr[zgr!=struct_height] = 10
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
        grid_df["elev"] = zgr[grid_df["idy"], grid_df["idx"]]
        print("  Done.")
        return grid_df, nesgr

    def setup_savepoints(self, grid_df):
        if self.save_pt_geojson == None:
            return None
        else:
            svpts = gpd.read_file(self.save_pt_geojson)
            grid_df = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.pt_x_wrld, grid_df.pt_y_wrld), crs=self.local_epsg)
            
            svpts = gpd.sjoin_nearest(svpts, grid_df[["idx","idy", "geometry"]], how="left", distance_col="distance")
            return svpts

    def remove_elevated_bldgs(self, bldgs, spatial_join=True):
        df = pd.read_csv(self.path_to_ffe)        
        if spatial_join == True:

            gdf_ffe = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="epsg:4326")
            gdf_ffe.to_crs(self.local_epsg, inplace=True)
            
            ffe_elev_status = []
            ffe_foundation = []
            for bldg_i, bldg in bldgs.iterrows():
                pt_in_poly = bldg.geometry.contains(gdf_ffe.geometry)
                if pt_in_poly.sum()>0:
                    bldg_ffe = gdf_ffe.loc[pt_in_poly==True]
                    bldg_ffe = bldg_ffe.iloc[0]
                    ffe_elev_status.append(bldg_ffe["FFE_elev_status"])
                    ffe_foundation.append(bldg_ffe["FFE_foundation"])
                else:
                    ffe_elev_status.append(np.nan)
                    ffe_foundation.append(np.nan)


            bldgs["FFE_elev_status"] = ffe_elev_status
            bldgs["FFE_foundation"]  = ffe_foundation
        else:
            df = df[["TA_FolioID", "FFE_elev_status", "FFE_foundation"]]
            bldgs = pd.merge(bldgs, df, left_on="FolioID", right_on="TA_FolioID")

        remove_bldgs = (bldgs["FFE_elev_status"] == "elevated") & (bldgs["FFE_foundation"]=="Piles/Columns")
        bldgs = bldgs.loc[~remove_bldgs]

        n_removed = remove_bldgs.sum()
        return bldgs


    def get_wind_angles(self, x, y):
        """
        Calculates the angle of a 2D vector from the x-axis in a counter-clockwise direction
        and returns it in the range [0, 360) degrees.
        """
        angles_rad = np.arctan2(y, x)
        angles_deg = np.degrees(angles_rad)
        # Convert negative angles to the [0, 360) range
        angles_deg[angles_deg < 0] += 360
        return angles_deg

    def setup_forcing(self, grid_df):
        """ first reading in water elevations data from xbeach*.out.
            if using swan spectra, then we delete the wave information 
            from this dataframe.
        """
        print("Preparing Forcing: ")
        forcing_dict = {}
        for uid_i, uid in enumerate(self.frcng_ids):
            frcng_pt_gdf_ = self.frcng_pts_gdf.loc[uid]

            fn = os.path.join(self.path_to_forcing, frcng_pt_gdf_["xbeach_file"])
            df_ = self.frcing_to_dataframe(fn, t_start=self.t_start, t_stop=self.t_stop)
            df_["windv"] = np.sqrt(np.square(df_["wx"]) + np.square(df_["wy"]))
            windth = self.get_wind_angles(df_["wx"].values, df_["wy"].values)
            windth_nautical = []
            for w_ in windth:
                w_ = self.cartesian_to_nautical_angle(w_)
                w_ = self.nautical_to_xbeach_angle(w_, self.alfa)
                windth_nautical.append(w_)
            df_["windth"] = windth_nautical
            if self.xbeach_params["wbctype"] == "swan":
                del df_["Hs"]
                del df_["Tp"]

            forcing_dict[uid] = df_.copy()

        # -- winds
        # writing water elevation data to output file
        wind_df = forcing_dict[self.frcng_ids[0]][["t_sec", "windv", "windth"]]

        fn_out = os.path.join(self.path_to_model, self.xbeach_params["windfile"])
        wind_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')
        
        # -- water elevations --
        """
            Note that XBeach goes clockwise around domain,
            Don provides ADCIRC water elevation data as below. I've re-named
             the xbeach*.out files such that they aling with the SWAN waves.
             That is, 
                - xbeach4.out has been renamed to xbeach3-nw.out and
                - xbeach3.out has been renamed to xbeach4-ne.out.
            The two offshore points for processing the forcing data are 
            therefore pts 1 and 3. 
            The two bayside points are 2 and 4

        """
        elev_df = pd.DataFrame()
        elev_df["t_sec"] = forcing_dict[self.frcng_ids[0]]["t_sec"]
        for uid_i, uid in enumerate(self.frcng_ids):
            elev_df["el_{}" .format(uid)] = forcing_dict[uid]["el"]
            if uid_i + 1 == self.xbeach_params["tideloc"]:
                break

        # writing water elevation data to output file
        fn_out = os.path.join(self.path_to_model, self.xbeach_params["zs0file"])
        elev_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')

        """ now turning attention to wave forcing.
            
            wave forcing can be specified at specific points, so I'm 
            using the forcing points geodataframe (frcg_pts_gdf) and the
            grid dataframe (grid_df) to determine which grid cell each 
            spectra is located within.

            The offshore points are swan points 1 and 3
            The bayside points are swan points 2 and 4. 
            See diagram above. 
        """
        grid_df_temp = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df["pt_x_wrld"], grid_df["pt_y_wrld"]), crs=self.local_epsg)
        frcg_pts_gdf = gpd.sjoin_nearest(self.frcng_pts_gdf, grid_df_temp[["idx", "idy", "pt_x", "pt_y", "geometry"]], how="left", distance_col="distance")

        if self.xbeach_params["wbctype"] == "swan":     # if wave forcing from swan spectra
            swan_points = self.frcng_pts_gdf.loc[self.frcng_ids]["swan_order"].to_list()
            s_, e_ = self.process_swan_output(n_header=97, n_locs=4, swan_points=swan_points, t_start=self.t_start, t_stop=self.t_stop)
            
            # # determine which swan spectra points to use. 
            # if self.xbeach_params["nspectrumloc"] == 1:
            #     spectra_points = [1]
            # elif self.xbeach_params["nspectrumloc"] == 2:
            #     spectra_points = offshore_points

            # if self.microdomain:
            #     spectra_points = [5]        # nearshore spectra

            for sp in swan_points:  # looping through wave spectra points and write to filelist. 
                fn_out = os.path.join(self.path_to_model, "filelist{}.txt" .format(sp))
                with open(fn_out, 'w') as f:
                    f.write("FILELIST\n")
                    for i in range(s_, e_):
                        fn = "swanpt{}-t{}.out\n" .format(sp, i)
                        f.write("900 0.5 {}" .format(fn))

            # now writing loclist
            fn = os.path.join(self.path_to_model, "loclist.txt")
            with open(fn, 'w') as f:
                f.write("LOCLIST\n" .format(self.model_name))
                for sp in swan_points:
                    idx = frcg_pts_gdf.loc[frcg_pts_gdf["swan_order"]==sp]["pt_x"].item()
                    idy = frcg_pts_gdf.loc[frcg_pts_gdf["swan_order"]==sp]["pt_y"].item()
                    f.write("{}. {}. filelist{}.txt\n" .format(idx, idy, sp))
            self.xbeach_params["dthetaS_XB"] = self.alfa

        elif self.xbeach_params["wbctype"]== "jonstable":   # if wave forcing from jonswap table
            if self.xbeach_params["nspectrumloc"] == 1:  # if one wave spectra provided
                sp_loc = 1
                wave_df = frcng_df[["Hs{}" .format(sp_loc), "Tp{}" .format(sp_loc), "mainang{}" .format(sp_loc), "gammajsp", "s", "duration", "dtbc"]]
                fn_out = os.path.join(self.path_to_model, self.xbeach_params["bcfile"])
                wave_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')

            else:                                       # if more than one wave spectra provided
                if self.xbeach_params["nspectrumloc"] == 2:
                    spectra_points = offshore_points
                elif self.xbeach_params["nspectrumloc"] == 4:
                    spectra_points = offshore_points + bayside_pts

                for sp in spectra_points:
                    Hs_key = "Hs{}" .format(sp)         # get Hs, Tp, and mainang for savepoint
                    Tp_key = "Tp{}" .format(sp)
                    mainang_key = "mainang{}" .format(sp)
                    wave_df = frcng_df[[Hs_key, Tp_key, mainang_key, "gammajsp", "s", "duration", "dtbc"]]

                    # write to jonswap file
                    fn_out = os.path.join(self.path_to_model, "jonswap{}.txt" .format(sp))
                    wave_df.to_csv(fn_out, sep="\t", index=None, header=None, float_format='%10.3f')

                fn = os.path.join(self.path_to_model, "loclist.txt")
                with open(fn, 'w') as f:
                    f.write("LOCLIST\n" .format(self.model_name))
                    for sp in spectra_points:
                        idx = frcg_pts_gdf.loc[frcg_pts_gdf["id"]==sp]["pt_x"].item()
                        idy = frcg_pts_gdf.loc[frcg_pts_gdf["id"]==sp]["pt_y"].item()
                        f.write("{}. {}. jonswap{}.txt\n" .format(idx, idy, sp))

        if self.drawfigs:
            cols = elev_df.columns
            cols = [i for i in cols if i!="t_sec"]
            for col in cols:
                fig, ax = plt.subplots(1,1)
                ax.plot(elev_df["t_sec"]/3600, elev_df[col])
                ax.set_title(col)
                if self.savefig:
                    fn = os.path.join(self.path_to_figs, "{}.png" .format(col))
                    plt.savefig(fn, 
                                transparent=False, 
                                dpi=500,
                                bbox_inches="tight",
                                pad_inches=0.1,
                                )
        print("  Done.")


    def frcing_to_dataframe(self, fn, n_header=3, n_var=7, t_start=0, t_stop=-1):
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
        
        if self.xbeach_params["wbctype"] != "swan":
            # ---
            # for testing
            # wavedir = 270
            # print("drs temporarily setting all angles to {}" .format(wavedir))
            # ---

            print("need to figureout s in jonswap params file")
            df["mainang"] = wavedir   # main wave angle
            df["gammajsp"] = 3.3    # peak enhancement factor for jonswap; not used in stationary mode
            df["s"] = 50          # directional spreading coeff (larger value results in longer wave crests)
            df["duration"] = dt   # duration of wave conditions
            df["dtbc"] = 0.5        # Timestep used to describe time series of wave energy and long wave flux at offshore boundary

        if t_start!=0:
            df = df.loc[df["t"]>=t_start]
            del df["t"]
            df["t"] = np.linspace(0, (len(df)-1)*dt/3600, len(df))
            df["t_sec"] = np.linspace(0, (len(df)-1)*dt, len(df))
        if t_stop != None:
            df = df.loc[df["t"]<=(t_stop-t_start)]

        return df
    
    def process_swan_output(self, n_header, n_locs, swan_points, t_start, t_stop):
        """
        function to process swan output from Don into appropriate format 
          for XBeach.
        Don provides swan spectra as single file at 7 save points and 385 
            time steps. XBeach needs the spectra for each save point and 
            time step as it's own file. 
        This function reads the swan spectra from Don, and writes out 
            individual spectra for each save point and time step. 

        Variables:
            n_header: number of header rows. Need to determine this manually. 
            n_locs: number of swan save points in original spectra file. need 
              determine manually.
            swan_points: which swan save points to use. See the point naming 
              convention and diagram in large comment above.
            t_start: what time step in SWAN time to start processing this data 
              from. For example, t_start=30 gets rid of first 30 hours of SWAN
              data.
            t_stop: time step to stop at in hours.
        Returns:
            t_start_step: this is the start step as an index.
            t_end_step: this is the end step as an index. 
        """
        fn_swan = os.path.join(self.path_to_forcing, "spts01.out")
        header_lines = []
        swan_spectra = {i: {} for i in range(n_locs)}
        swan_loc_cnt = n_locs - 1

        time = -1
        with open(fn_swan,'r') as f:
            for cnt, line in enumerate(f.readlines()):
                if cnt < n_header:
                    header_lines.append(line)
                    if ("number of locations" in line):
                        n_locs_check = int([x.strip() for x in line.split()][0])
                        if n_locs != n_locs_check:
                            raise ValueError("Number of locations in input and swan file do not match. ")
                else:
                    if ("date and time" in line):
                        curr_date_time = line
                        continue

                    if ("ZERO" in line) or ("FACTOR" in line) or ("NODATA" in line):
                        swan_loc_cnt += 1
                        if swan_loc_cnt == n_locs:
                            swan_loc_cnt = 0
                            time += 1
                        if (time in swan_spectra[swan_loc_cnt]) == False:
                            swan_spectra[swan_loc_cnt][time] = []
                    
                    swan_spectra[swan_loc_cnt][time].append(line)

        # each swan file represents 15 minutes.
        t_start_step = int(t_start*60/15)
        t_stop_step  = int(t_stop*60/15)
        for spectra in swan_points:
            n_swan_spectra = 0
            for time in range(t_start_step,t_stop_step):
                n_swan_spectra += 1
                if time == 0:
                    continue
                # fn_out = os.path.join(swan_dir_out, "swanpt{}-t{}.out" .format(spectra, time))
                fn_out = os.path.join(self.path_to_model, "swanpt{}-t{}.out" .format(int(spectra), time))
                latlong_written = False
                with open(fn_out, 'w') as f:
                    for l in header_lines:
                        if ("TIME" in l) or ("time coding option" in l):
                            continue
                        if ("number of locations" in l):
                            l = "1                                  number of locations\n"
                            f.write(l)
                            continue
                        if "-81." in l:
                            if latlong_written == False:
                                f.write("  0   0\n")
                            latlong_written = True
                            continue
                        f.write(l)
                    
                    for l in swan_spectra[spectra][time]:
                        f.write(l)

        return t_start_step, t_stop_step

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


    def create_model(self, grid_df, nesgr, savepoint_df):
        print("Writing model: ")
        xgr, ygr, zgr = self.grid_df_to_xyz(grid_df)
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
        
        if "tstop" not in self.xbeach_params:
            # self.xbeach_params["tstop"] = elev_df.iloc[-1]["t_sec"].astype(int)
            self.xbeach_params["tstop"] = (self.t_stop - self.t_start)*3600

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

        self.write_xbeach_params(savepoint_df)
        self.move_figs()
        print("  Done.")
        

    def rename_file(self, fn_old, fn_new):
        fn_old = os.path.join(self.path_to_model, fn_old)
        fn_new = os.path.join(self.path_to_model, fn_new)
        os.rename(fn_old, fn_new)


    def write_xbeach_params(self, savepoint_df):
        drs_input_keys = ["xo", "yo", "theta"] # keys for dylan to post-process, but not used by xbeach
        grid_input_keys = ["nx", "ny", "dx", "dy", "xori", "yori", "alfa", "depfile", "vardx", "xfile", "yfile", "posdwn", "thetamin", "thetamax", "dtheta",  "dtheta_s","dthetaS_XB", "wavint"]
        numerics_input_keys = ["CFL", "eps", "front", "back", "scheme", "cyclic", "left", "right", "maxdtfac", "mmpi", "nmpi", "mpiboundary", "outputprecision", "random"]
        time_input_keys = ["dt", "tstart", "tintg", "tintm", "tintp", "tstop", "taper", "dtset"]
        general_constants = ["rho", "g"]
        boundary_condition_keys = ["zs0file", "tideloc", "paulrevere", "tidetype", "tidelen", "zs0", "bcfile", "rt", "dtbc", "sprdthr", "wbcversion", "nspectrumloc", "wind", "windfile"]
        wave_calculation_keys = ["wavemodel", "wbctype", "instat", "break", "wci", "roller", "beta", "gamma", "gammax", "alpha", "delta", "n", "maxerror", "maxiter"]
        flow_calculation_keys = ["nuh", "nuhfac", "nuhv", "umin"]
        sed_trans_calculation_keys = ["sedtrans", "dico", "D50", "D90", "rhos", "z0"]
        morphological_calculation_keys = ["morphology", "struct", "ne_layer", "morfac", "morstart", "por", "dryslp", "wetslp", "hswitch"]

        fn = os.path.join(self.path_to_model, "params.txt")
        with open(fn, 'w') as f:
            f.write("XBeach Model : {}\n" .format(self.model_name))
            f.write("Params Written : {}\n" .format(datetime.now().replace(second=0, microsecond=0)))

            self.write_xbeach_params_section(f, "Post-Processing Grid Location (not used by XB)", drs_input_keys, sep=":")
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

            f.write("\n")
            if savepoint_df != None:
                f.write("{:20s} = {}\n" .format("npoints", len(savepoint_df)))
                for row_i, row in savepoint_df.iterrows():
                    f.write("{} {}\n" .format(row["idx"], row["idy"]))
                f.write("{:20s} = {}\n" .format("npointvar", len(self.xbeach_params["point_var"])))
                for i in self.xbeach_params["point_var"]:
                    f.write("{}\n" .format(i))


    def write_xbeach_params_section(self, f, title, key_list, sep="="):
            f.write("\n-----------------------------------------\n" )
            f.write("\n{}\n" .format(title))
            for key in key_list:
                if key in self.xbeach_params.keys():
                    f.write("{:20s} {} {}\n" .format(key, sep, str(self.xbeach_params[key])))



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
        returns a numpy array with dimensions equal to the size of 
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
        if self.microdomain:
            w = sides[np.argsort(sides)][2:]        # width  is defined here as crossshore distance
            l = sides[np.argsort(sides)][0:2]       # length is defined here as alongshore distance

        else:
            # note: the two lines below are used for large domain.
            w = sides[np.argsort(sides)[0:2]]       # width  is defined here as crossshore distance
            l = sides[np.argsort(sides)[2:]]        # length is defined here as alongshore distance
        
        theta_r, theta_d = self.compute_theta((xo, yo), (xa, ya))   # angle that grid is rotated, measured relative to east
        self.origin = (xo, yo)
        self.alfa = theta_d         # storing angle of grid rotation to self.alfa



        l = self.myround(np.average(l), base=self.xbeach_params["xbeach_res"])
        w = self.myround(np.average(w), base=self.xbeach_params["xbeach_res"])
        nx = int(w/self.xbeach_params["xbeach_res"])
        ny = int(l/self.xbeach_params["xbeach_res"])

        # -- new
        # -- setting up x and y as x-y points (0, 1, 2, 3)
        x = np.arange(0, nx)    # 
        y = np.arange(0, ny)
        
        # -- setting up grid of indicies; flattening to put in grid_df
        idx, idy = np.meshgrid(x, y)
        idx, idy = idx.flatten(), idy.flatten()

        # -- setting up pt_x and pt_y; this is with XBeach resolution
        pt_x = x*self.xbeach_params["xbeach_res"]
        pt_y = y*self.xbeach_params["xbeach_res"]

        # -- setting up grid of pt_x and pt_y; flattening to put in grid_df
        pt_x, pt_y = np.meshgrid(pt_x, pt_y)
        pt_x, pt_y = pt_x.flatten(), pt_y.flatten()

        # -- setting x and y coordinates in real-world points; uses origin above which is in model crs
        pt_x_wrld = xo + pt_x*np.cos(theta_r) - pt_y*np.sin(theta_r)
        pt_y_wrld = yo + pt_x*np.sin(theta_r) + pt_y*np.cos(theta_r)

        # -- loop through coordinates and get elevations
        coord_list = [(x, y) for x, y in zip(pt_x_wrld, pt_y_wrld)]
        with rasterio.open("temp.tiff", "r") as src:
            elev = [x[0].item() for x in src.sample(coord_list)]
        
        # setting up grid_df
        grid_df = pd.DataFrame()
        grid_df["pt_x"] = pt_x
        grid_df["pt_y"] = pt_y
        grid_df["pt_x_wrld"] = pt_x_wrld
        grid_df["pt_y_wrld"] = pt_y_wrld
        grid_df["idx"] = idx
        grid_df["idy"] = idy
        grid_df["elev"] = elev

        grid = np.zeros((nx,ny))
        grid[grid_df["idx"], grid_df["idy"]] = grid_df["elev"]
        
        # --
        if self.smooth_grid_tf:
            grid = self.smooth_grid(grid)
        if self.extend_grid_tf:
            if self.xbeach_params["cyclic"] == 1:
                grid, grid_df = self.repeat_grid_bottom(grid, grid_df, xo, yo, theta_r, pad=250)
                grid, grid_df = self.expand_grid_top(grid, grid_df, xo, yo, theta_r, pad=250)
            else:
                grid, grid_df = self.expand_grid_bottom(grid, grid_df, xo, yo, theta_r, pad=200)
                grid = self.smooth_grid(grid)   # need to resmooth after expanding in bottom. I end up resampling elev when not using cyclic BCs

        grid_df["elev"] = grid[grid_df["idx"], grid_df["idy"]]

        # formatting data to pass back
        xvals = grid_df["pt_x"].unique()
        yvals = grid_df["pt_y"].unique()


        # -- storing x,y origin and angle of rotation; not used by xbeach
        self.xbeach_params["xo"] = grid_df.loc[(grid_df["pt_x"]==0)&(grid_df["pt_y"]==0), "pt_x_wrld"].item()
        self.xbeach_params["yo"] = grid_df.loc[(grid_df["pt_x"]==0)&(grid_df["pt_y"]==0), "pt_y_wrld"].item()
        self.xbeach_params["theta"] = theta_d
        
        # self.write_grid(grid_df)
        self.xbeach_params["nx"] = len(xvals) - 1
        self.xbeach_params["ny"] = len(yvals) - 1
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



