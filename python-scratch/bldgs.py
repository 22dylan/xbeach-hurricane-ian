import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

class bldgs(object):
    """docstring for bldgs"""
    def __init__(self):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        path_to_buildings = os.path.join(self.file_dir, "..", "data", "buildings", "ft_myers_bldgs_micro.geojson")
        
        self.ms_bldgs = self.read_bldgs(path_to_buildings)
        self.crs = self.ms_bldgs.crs


    def read_bldgs(self, fn):
        gdf = gpd.read_file(fn)
        return gdf


    def merge_ffe(self):
        path_to_ffe = os.path.join(self.file_dir, "..", "data", "mehrshad", "data", "Geoscience-collection--overall-dataset", "data", "FMB_VDA_FFE_Final.csv")
        df = pd.read_csv(path_to_ffe)

        gdf_ffe = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="epsg:4326")
        gdf_ffe.to_crs(self.crs, inplace=True)
        cnt = 0
        cols = ["VDA_id", "FFE_elev_status", "FFE_ffe_ft", "FFE_lhsm_ft"]
        
        vda_id = []
        ffe_elev_status = []
        ffe_ft = []
        ffe_lhsm_ft = []
        ffe_foundation = []

        for bldg_i, bldg in self.ms_bldgs.iterrows():
            pt_in_poly = bldg.geometry.contains(gdf_ffe.geometry)
            if pt_in_poly.sum()>0:
                bldg_ffe = gdf_ffe.loc[pt_in_poly==True]
                # if len(bldg_ffe)>1:
                bldg_ffe = bldg_ffe.iloc[0]

                vda_id.append(bldg_ffe["VDA_id"])
                ffe_elev_status.append(bldg_ffe["FFE_elev_status"])
                ffe_ft.append(bldg_ffe["FFE_ffe_ft"])
                ffe_lhsm_ft.append(bldg_ffe["FFE_lhsm_ft"])
                ffe_foundation.append(bldg_ffe["FFE_foundation"])

                cnt += pt_in_poly.sum()
                
            else:
                vda_id.append(np.nan)
                ffe_elev_status.append(np.nan)
                ffe_ft.append(np.nan)
                ffe_lhsm_ft.append(np.nan)
                ffe_foundation.append(np.nan)

        self.ms_bldgs["vda_id"] = vda_id
        self.ms_bldgs["ffe_elev_status"] = ffe_elev_status
        self.ms_bldgs["ffe_ft"] = ffe_ft
        self.ms_bldgs["ffe_lhsm_ft"] = ffe_lhsm_ft
        self.ms_bldgs["ffe_foundation"] = ffe_foundation

        df = pd.DataFrame(self.ms_bldgs)

        del df["geometry"]
        df.to_csv("temp.csv")
        remove_bldgs = (self.ms_bldgs["ffe_elev_status"] == "elevated") & (self.ms_bldgs["ffe_foundation"]=="Piles/Columns")
        self.ms_bldgs = self.ms_bldgs.loc[~remove_bldgs]
        self.ms_bldgs.to_file("merged_gdf.geojson")


if __name__ == '__main__':
    b = bldgs()
    b.merge_ffe()
