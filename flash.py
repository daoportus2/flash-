# -*- coding: utf-8 -*-
"""
Created on Mon May 19, 2025

Author: oportus.d

This script extracts and visualizes 1D spatial profiles (Y-direction) from FLASH 2D simulation outputs,
averaging along the X-direction over time. It supports density, temperature, pressure, and bremsstrahlung power.

Ensure the 'plasma.py' module is either in the same folder or add its directory using:
    import sys
    sys.path.append('C:/Users/oportus.d/Documents/PythonCodes/')
    import plasma as p
"""

import os
import sys
import numpy as np
import yt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from glob import glob

# === Add custom module paths ===
sys.path.append('C:/Users/oportus.d/Documents/RadMag_model/Radiative_codes/')
sys.path.append('C:/Users/oportus.d/Documents/PythonCodes/')

# === Import custom physics modules ===
import plasma as p  # Required for brem calculations




def sop1d_Flash(MasterPath, field='tele', resolution=200, width_x=150e-4, x_center=0,
                normalize=False, label='output', cmap='inferno'):
    """
    Process FLASH simulation outputs to extract a 2D (Y vs. time) profile
    of a given field, averaged over a finite X-width.

    Parameters
    ----------
    MasterPath : str
        Directory containing FLASH `plt_cnt` output files.
    field : str
        Field to extract: 'tele', 'dens', 'pressure', or 'brem'.
    resolution : int
        Resolution along each axis (both X and Y).
    width_x : float
        X-width (in cm) over which to average the field.
    x_center : float
        X-center (optional, overridden from data).
    normalize : bool
        Not used; placeholder for future normalization.
    label : str
        Label for output files and plots.
    cmap : str
        Colormap for the space-time diagram.

    Returns
    -------
    density_matrix2d : np.ndarray
        2D array of field values.
    time_array : np.ndarray
        Time array [ns].
    y_array : np.ndarray
        Spatial Y-axis [microns].
    """

    # Use folder name as label if not specified
    label = os.path.basename(os.path.normpath(MasterPath))

    # Output directories
    MasterPathOut_img = os.path.join(MasterPath, 'img_sop1d')
    MasterPathOut_np_matrix = os.path.join(MasterPath, 'np_matrix_sop1d')
    os.makedirs(MasterPathOut_img, exist_ok=True)
    os.makedirs(MasterPathOut_np_matrix, exist_ok=True)

    # Collect FLASH plot files (ignore initial _0000)
    file_list = sorted([
        f for f in glob(os.path.join(MasterPath, "lasslab_hdf5_plt_cnt_*"))
        if os.path.isfile(f) and "0000" not in os.path.basename(f)
    ])

    # Initialize storage containers
    density_matrix = []
    time_list = []

    res = [resolution, resolution]
    ny = res[1]

    # Loop over each FLASH file
    for file in file_list:
        ds = yt.load(file)
        time_ns = ds.current_time.to("ns").v

        # Domain bounds
        left_edge = ds.domain_left_edge
        right_edge = ds.domain_right_edge

        # Compute X-slice range for averaging
        x_center = 0.5 * (left_edge[0] + right_edge[0])
        x_left = x_center - 0.5 * width_x
        x_right = x_center + 0.5 * width_x

        # Determine max location in slice plane
        if field in ['pressure', 'brem']:
            _, c = ds.find_max(("flash", 'pres'))
        else:
            _, c = ds.find_max(("flash", field))

        # Slice and project
        width = ds.domain_width[0]
        sli = ds.slice(2, c[2])
        frb = sli.to_frb(width, res)

        # === Convert to matrix based on selected field ===
        if field == 'dens':
            units = 'g/cm**3'
            matrix = np.array(frb[field].to(units))

        elif field == 'tele':
            units = 'K'
            matrix = np.array(frb[field].to(units))

        elif field == 'pressure':
            units = 'Pa'
            matrix = (
                np.array(frb['pele'].to(units)) +
                np.array(frb['pion'].to(units)) +
                np.array(frb['prad'].to(units))
            )

        elif field == 'brem':
            units = 'W/m**3'
            n_e = 6.02e23 * np.array(frb['dens'].to('g/cm**3')) * np.array(frb['ye']) * 1e6
            T_e = np.array(frb['tele'].to('K'))
            Z_eff = np.array(frb['ye']) / np.array(frb['sumy'])
            matrix = p.radiation_power(n_e, T_e, Z_eff)

        # Extract averaged Y-profile across selected X-range
        x_min, x_max, y_min, y_max = frb.bounds
        x_lin = np.linspace(x_min, x_max, res[0])
        ix_left = np.searchsorted(x_lin, x_left)
        ix_right = np.searchsorted(x_lin, x_right)

        matrix_slide = matrix[:, ix_left:ix_right]
        y_profile = np.mean(matrix_slide, axis=1)

        # Append current profile and time
        density_matrix.append(y_profile)
        time_list.append(time_ns)

    # === Assemble time-series 2D matrix ===
    y_axis_vals = np.linspace(float(left_edge[1]), float(right_edge[1]), ny) * 1e4  # cm → µm
    density_matrix2d = np.vstack(density_matrix)
    time_array = np.array(time_list)
    y_array = np.array(y_axis_vals)

    # Save matrices
    np.save(os.path.join(MasterPathOut_np_matrix, f'matrix_{field}_{label}_sop1d_unstack.npy'), density_matrix)
    np.save(os.path.join(MasterPathOut_np_matrix, f'matrix2d_{field}_{label}_sop1d_unstack.npy'), density_matrix2d)
    np.save(os.path.join(MasterPathOut_np_matrix, f'time_array_{label}_sop1d_unstack.npy'), time_array)
    np.save(os.path.join(MasterPathOut_np_matrix, f'y_array_{label}_sop1d_unstack.npy'), y_array)

    # Optional: convert temperature to eV
    if field == 'tele':
        density_matrix2d /= 11604,5250061657
        units = 'eV'
    else:
        units = units

    # === Plot space-time diagram ===
    plt.figure(figsize=(10, 6))
    extent = [y_array.min(), y_array.max(), time_array.min(), time_array.max()]

    if field != 'tele':
        plt.imshow(density_matrix2d, aspect='auto', extent=extent, origin='lower',
                   cmap=cmap, norm=LogNorm(vmin=max(density_matrix2d.min(), 1e-3), vmax=density_matrix2d.max()))
    else:
        plt.imshow(density_matrix2d, aspect='auto', extent=extent, origin='lower',
                   cmap=cmap, vmin=np.percentile(density_matrix2d, 1), vmax=np.percentile(density_matrix2d, 98))

    plt.gca().invert_yaxis()
    field_display = 'Power Bremsstrahlung' if field == 'brem' else field
    plt.colorbar(label=f'{field_display} [{units}]')
    plt.xlabel('Y [µm]')
    plt.ylabel('Time [ns]')
    plt.title(f'{field_display} evolution along Y (avg over {int(width_x * 1e4)} µm in X), {label}')
    plt.tight_layout()

    # Save figure
    output_img = os.path.join(MasterPathOut_img, f'{field}_{label}_sop1d.png')
    plt.savefig(output_img, dpi=300)
    plt.show()

    print(f'Matrices saved in: {MasterPathOut_np_matrix}')
    print(f'Image saved in: {output_img}')

    return density_matrix2d, time_array, y_array
