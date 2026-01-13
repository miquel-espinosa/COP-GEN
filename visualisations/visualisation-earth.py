import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def generate_global_grid_plot():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(10, 30))
    
    # Add land, ocean, and country boundaries
    ax.add_feature(cfeature.LAND, color='bisque', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightskyblue', alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='grey', linewidth=0.75)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.gridlines(draw_labels=False, linewidth=1, linestyle='dotted', color='black', 
                xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-90, 91, 30))
    
    # Generate equidistant points using Fibonacci lattice
    num_points = 20000  # Number of points to distribute
    
    # Golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    # Generate the points
    i = np.arange(0, num_points)
    z = 1 - (2*i + 1) / num_points  # z ranges from 1 to -1
    radius = np.sqrt(1 - z**2)      # radius at each height
    
    # Calculate longitude based on golden angle
    theta = golden_angle * i
    
    # Convert to Cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Convert to lat/lon for plotting (degrees)
    lat = np.arcsin(z) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    
    # Plot the points
    ax.scatter(lon, lat, color='tomato', s=2, transform=ccrs.PlateCarree(), alpha=0.5)
    
    plt.savefig('global_grid.png', dpi=300)

# Run the function
generate_global_grid_plot()
