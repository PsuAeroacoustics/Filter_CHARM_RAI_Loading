#!/usr/bin/env python3

import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))

from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  

import plot_styles
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import cmcrameri.cm as cmc
# import cmap as cm

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.',':']
marker = ['o','^','*']

case_name = "quickROD_DN05_SDOF_DIST_OAR15_OPT/quickROD.1PSU-WOPWOP"
R = 0.4699/2
case_dir = os.path.join(os.getcwd())
acs_data ={}

if not os.path.exists(os.path.join(case_dir,f'{case_name}.h5')):
    process_wopwop(cases_directory=os.path.join(case_dir,case_name),cases = 'cases.nam')

acs_data = import_results_from_wopwop(cases_directory=os.path.join(case_dir,case_name))
theta = np.arctan2(acs_data['geometry_values'][:,:,0,1],acs_data['geometry_values'][:,:,0,0])%(2*np.pi)
theta[-1] = 2*np.pi
phi = np.abs(np.arctan2(acs_data['geometry_values'][:,:,0,-1],np.linalg.norm((acs_data['geometry_values'][:,:,0,0],acs_data['geometry_values'][:,:,0,1]),axis = 0)))
r = np.linalg.norm(acs_data['geometry_values'][:,:,0],axis = -1).mean()

#%%

class Arrow3D(FancyArrowPatch):
    """3D arrow patch for matplotlib."""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = np.array([x, y, z])
        self._dxdydz = np.array([dx, dy, dz])

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        xs, ys, zs = proj_transform((x1, x1+dx), (y1, y1+dy), (z1, z1+dz), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        xs, ys, zs = proj_transform((x1, x1+dx), (y1, y1+dy), (z1, z1+dz), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)


def plot_sphere_with_coordinates(radius=1.0, azimuth_angle=45, elevation_angle=45, figsize=(8, 8)):
    """
    Create a 3D plot of a sphere with coordinate system indicators.
    
    Parameters:
    -----------
    radius : float
        Radius of the sphere (default: 1.0)
    azimuth_angle : float
        Azimuthal angle for radial arrow endpoint (in degrees, default: 45)
    elevation_angle : float
        Elevation angle for radial arrow endpoint (in degrees, default: 45)
    figsize : tuple
        Figure size (default: (10, 10))
    """
    
    fig = plt.figure(figsize=figsize,constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(right=0.85)
    # Create sphere surface
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot sphere surface
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.15, color='gray', edgecolor='none')
    
    # Convert angles to radians
    phi = np.radians(azimuth_angle)  # azimuthal angle
    theta = np.radians(elevation_angle)  # elevation angle from zenith
    
    # Radial arrow endpoint (in spherical coordinates)
    x_rad = radius * np.sin(theta) * np.cos(phi)
    y_rad = radius * np.sin(theta) * np.sin(phi)
    z_rad = radius * np.cos(theta)
    
    # Draw radial arrow from center to surface
    arrow_radial = Arrow3D(0, 0, 0, x_rad, y_rad, z_rad,
                          mutation_scale=8, lw=2.5, arrowstyle="-|>", color="black")
    ax.add_artist(arrow_radial)
    
    # --- Azimuthal direction arrow (phi direction) ---
    # This arrow curves in the x-y plane (z=0, theta=90 degrees)
    phi_start = np.radians(0)
    phi_end = np.radians(azimuth_angle)
    phi_curve = np.linspace(phi_start, phi_end, 30)
    
    # Radius of circular arc in xy-plane at equator
    r_xy = radius*.6  # at theta = 90 degrees, sin(theta) = 1
    
    x_phi = r_xy * np.cos(phi_curve)
    y_phi = r_xy * np.sin(phi_curve)
    z_phi = np.zeros_like(phi_curve)  # z=0 for x-y plane
    
    ax.plot(x_phi, y_phi, z_phi, 'green', linewidth=2.5, label='Positive')
    
    # Add arrowhead for azimuthal direction
    arrow_phi = Arrow3D(x_phi[-1], y_phi[-1], z_phi[-1],
                       x_phi[-1] - x_phi[-2], y_phi[-1] - y_phi[-2], z_phi[-1] - z_phi[-2],
                       mutation_scale=8, lw=2.5, arrowstyle="-|>", color="green")
    ax.add_artist(arrow_phi)
    
    # Add label for azimuthal angle
    label_distance = r_xy * 1.1
    phi_label = np.radians(azimuth_angle+5)
    x_phi_label = label_distance * np.cos(phi_label/2) 
    y_phi_label = label_distance * np.sin(phi_label/2)
    z_phi_label = -2
    ax.text(x_phi_label, y_phi_label, z_phi_label, r'$+\psi$', color='green',fontsize=14, fontweight='bold')
    
    # --- Elevation direction arrow (theta direction) ---
    # This arrow curves from the x-y plane (theta=90°) to the specified elevation
    theta_start = np.radians(90)  # Start from x-y plane
    theta_end = np.radians(elevation_angle)
    theta_curve = np.linspace(theta_start, theta_end, 30)
    
    x_theta = r_xy * np.sin(theta_curve) * np.cos(phi)
    y_theta = r_xy * np.sin(theta_curve) * np.sin(phi)
    z_theta = r_xy * np.cos(theta_curve)
    
    ax.plot(x_theta, y_theta, z_theta, 'purple', linewidth=2.5, label='Positive')
    
    # Add arrowhead for elevation direction
    arrow_theta = Arrow3D(x_theta[-2], y_theta[-2], z_theta[-2],
                         x_theta[-1] - x_theta[-2], y_theta[-1] - y_theta[-2], z_theta[-1] - z_theta[-2],
                         mutation_scale=8, lw=2.5, arrowstyle="-|>", color="purple")
    ax.add_artist(arrow_theta)
    
    # Add label for elevation angle
    theta_label_rad = np.radians(50+elevation_angle/2)
    x_theta_label = label_distance * np.sin(theta_label_rad) * np.cos(phi_label) 
    y_theta_label = label_distance * np.sin(theta_label_rad) * np.sin(phi_label) 
    z_theta_label = label_distance * np.cos(theta_label_rad)
    ax.text(x_theta_label, y_theta_label, z_theta_label, r'$+\phi$', fontsize=14,color='purple')
    
    # Add coordinate axes at origin
    axis_length = radius * 1.3
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=1.5, alpha=0.6, label='X-axis')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, linewidth=1.5, alpha=0.6, label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, linewidth=1.5, alpha=0.6, label='Z-axis')
    
    # Plot the center point
    ax.scatter(acs_data['geometry_values'][:,:,0,0]/R, acs_data['geometry_values'][:,:,0,1]/R, acs_data['geometry_values'][:,:,0,2]/R, color='red', s=40, marker='o')
    
    # Plot the point on sphere surface
    ax.scatter([0], [0], [0], color='black', s=10, marker='o')
    
    # Set labels and titles
    ax.set_xlabel('x/R')
    ax.set_ylabel('y/R')
    ax.set_zlabel('z/R')
    # ax.set_title(f'Sphere (r={radius}) with Spherical Coordinates\n' + 
    #              f'Azimuth φ={azimuth_angle}°, Elevation θ={elevation_angle}°',
    #              fontsize=14, fontweight='bold', pad=20)
    
    # Set equal aspect ratio
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_zlim([-8, 8])

    ax.grid(True)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.15)

    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
        
    # Set viewing angle
    ax.view_init(elev=25, azim=55,roll = -2)
    # ax.set_position([0.1, 0.1, 0.9, 0.8])

    plt.tight_layout()
    
    return fig, ax

fig, ax = plot_sphere_with_coordinates(radius=r/R, azimuth_angle=35, elevation_angle=50,figsize=(4,4))
plt.savefig(os.path.join(case_dir,'obs_sphere.pdf'),format = 'pdf',pad_inches=.4,bbox_inches='tight')

print('done')