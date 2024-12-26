# -*- coding: utf-8 -*-
'''
This is the demo file to show you how to use SUVEL

Written by Dr. Jiajia Liu @ University of Science and Technology of China

Cite Liu et al. 2025, Tracking the Photospheric Horizontal Velocity Field with Shallow U-Net Models

Revision History
2024.12.20
    Version 1.0 - Initial release

'''

import numpy as np
from suvel import suvel
from utils import plot_comparison


# First of all, read the data
suvel_path = './'
data = dict(np.load('demo_data.npz', allow_pickle=True))

# Normlized photospheric intensity, with a size of [ny=512, nx=512, nt=3]
# The last dimension has 3 elements, they are the intensity at t-dt, t, and t+dt
# dt is the cadence of the data
intensity = data['intensity']

# Normalized photospheric vertical magnetic field, with as size of [ny=512, nx=512, nt=3]
magnetic = data['magnetic']

# Original velocity field, with a size of [ny=512, nx=512]
vx = data['vx']
vy = data['vy']
v = np.stack((vx, vy), axis=2)

# ds (pixel size) in units of km, and dt (cadence) in units of second
ds = data['ds']
dt = data['dt']

# do prediction using intensity only
vi = suvel(intensity=intensity, model_path=suvel_path)

# do prediction using magnetic field only
vm = suvel(magnetic=magnetic, model_path=suvel_path)

# do prediction using both
vh = suvel(intensity=intensity, magnetic=magnetic, model_path=suvel_path)

# scale back to pixel per frame
vi = vi  * 20 - 10.
vm = vm * 20 - 10.
vh = vh * 20 - 10.

# scale back to km/s
vi = vi * ds / dt
vm = vm * ds / dt
vh = vh * ds / dt

# show comparison for the intensity model
plot_comparison(v, vi, fig_num=0, ds=ds, ds_unit='km', subscript='i',
                vmin=-10, vmax=10, vunit='km/s')

# show comparison for the magnetic model

plot_comparison(v, vm, fig_num=1, ds=ds, ds_unit='km', subscript='m',
                vmin=-10, vmax=10, vunit='km/s')

# show comparison for the hybrid model

plot_comparison(v, vh, fig_num=2, ds=ds, ds_unit='km', subscript='h',
                vmin=-10, vmax=10, vunit='km/s')