# import os
# import geopandas as gpd
# import matplotlib.pyplot as plt

# fn = os.path.join(os.getcwd(), "..", "data", "buildings", "ft_myers_bldgs_estero.geojson")
# gdf_buildings = gpd.read_file(fn)

# fig, (ax0, ax1) = plt.subplots(1,2, figsize=(16,9))
# gdf_buildings.plot(ax=ax0, zorder=1, facecolor="grey", edgecolor="red", linewidth=2)
# xlim = [409607, 409933]
# ylim = [2922763, 2923416]
# ax0.set_xlim(xlim)
# ax0.set_ylim(ylim)

# gdf_buildings["geometry"] = gdf_buildings["geometry"].convex_hull
# gdf_buildings["geometry"] = gdf_buildings["geometry"].simplify(tolerance=5)
# gdf_buildings["geometry"] = gdf_buildings["geometry"].offset_curve(distance=-1)

# # gdf2 = gdf_buildings.copy()
# # gdf2["geometry"] = gdf_buildings["geometry"].convex_hull
# # print(gdf2.head())

# gdf_buildings.plot(ax=ax1, zorder=1, facecolor="grey", edgecolor="red", linewidth=2)
# ax1.set_xlim(xlim)
# ax1.set_ylim(ylim)

# plt.show()


# # loop through each buildling and finds points in building geom; change elevation to 99
# for cnt, bldg_ in gdf_buildings.iterrows():
#     print(bldg_.)
#     fds
# for i in range(len(gdf_buildings)):
#     bldg_ = gdf_buildings.iloc[i]
#     gdf_temp = bldg_.geometry.contains(grid_df.geometry)
#     if gdf_temp.sum()>0:        # if there is a grid cell with a building on.
#         grid_ = grid_df.loc[gdf_temp]
#         zgr[grid_["idy"], grid_["idx"]] = struct_height



import numpy as np
from scipy.ndimage import binary_opening, generate_binary_structure

# 1. Create a sample 2D NumPy array
# This array has a large group of 99999s, a jutting arm, and an isolated point.
zgro = np.array([
    [10, 20, 30, 40, 50, 60],
    [70, 104, 1934, 13, 14, 110],
    [120, 14, 14, 142, 150, 160],
    [170, 432, 242, 200, 210, 220],
    [230, 240, 250, 42, 270, 280],
    [10, 20, 30, 40, 50, 60],
    [70, 104, 1934, 13, 14, 110],
    [120, 14, 14, 142, 150, 160],
    [170, 432, 242, 200, 210, 220],
    [230, 240, 250, 42, 270, 280],
    [290, 300, 310, 320, 330, 340]
    ])

zgr = np.array([
    [10,  20,    30,    40,    50,    60],
    [70,  104,   1934,  13,    14,    110],
    [120, 14,    14,    142,   150,   160],
    [170, 432,   99999, 99999, 99999,   220],
    [230, 240,   99999, 99999, 99999,   280],
    [10,  20,    99999, 99999, 99999,    60],
    [70,  99999, 99999, 99999, 99999, 110],
    [120, 99999, 99999, 99999, 150,   160],
    [170, 99999, 99999, 99999, 210,   220],
    [230, 240,   250,   99999, 270,   280],
    [290, 300,   310,   320,   330,   340]
])

print("Original Array:")
print(zgr)

# 2. Create a boolean mask where 99999 is True
mask = (zgr == 99999)

# 3. Define the structuring element for the morphological operation.
# A 3x3 square is a common choice for a 2D array.
# 'generate_binary_structure' creates a connectivity mask.
struct = generate_binary_structure(2, 2)

# 4. Perform the binary opening operation.
# The `iterations=1` parameter is usually sufficient to remove small artifacts.
cleaned_mask = binary_opening(mask, structure=struct, iterations=1)

# 5. Create a new array, applying the cleaned mask.
# We find the values that were 99999 in the original array but are now False in the cleaned mask.
# These are the isolated points and juts. We replace them with a non-99999 value (e.g., 0).
cleaned_data = zgr.copy()
cleaned_data[mask & ~cleaned_mask] = zgro[mask & ~cleaned_mask]

print("\nCleaned Array (Juts and isolated points are replaced with 0):")
print(cleaned_data)
print("\n\n")

# # You can also get a mask of just the remaining, large groups
# final_mask = cleaned_data == 99999
# print("\nFinal Mask (only the large, connected group):")
# print(final_mask.astype(int))


