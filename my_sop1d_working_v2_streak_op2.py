# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:01:38 2025

@author: oportus.d
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import yt
from yt import YTQuantity
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, eV
from scipy.constants import Boltzmann as kb
from scipy.constants import c as c
from scipy.constants import Planck as h
from matplotlib.colors import LogNorm
from glob import glob
from scipy.constants import m_e

from scipy.constants import e as e_ch
#%%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
#%%
def find_nearest_post_peak(y_dense, row_smoothed, target_density):
    # Step 1: Find peak index
    peak_idx = np.argmax(row_smoothed)
    
    # Step 2: Slice after the peak
    y_tail = y_dense[peak_idx:]
    row_tail = row_smoothed[peak_idx:]

    # Step 3: Find index in tail closest to target
    if len(row_tail) == 0:
        return np.nan  # edge case: empty tail

    idx_nearest = np.argmin(np.abs(row_tail - target_density))
    
    # Step 4: Get corresponding y value
    return idx_nearest,y_tail[idx_nearest]
#%%

def interpolate_position_from_density(row, y_array, target_density):

    
    # Check bounds
    target_closest = find_nearest(row,target_density )
    idx_closest = np.where(row == find_nearest(row,target_density ))[0][0]
    
    if target_closest<target_density:
        idx_above = idx_closest+1
        idx_below = idx_closest
    elif target_closest>target_density:
        idx_above = idx_closest
        idx_below = idx_closest+1
    
    # Find bounding indices
   

    rho0, rho1 = row[idx_below], row[idx_above]
    y0, y1 = y_array[idx_below],y_array[idx_above]

    # Linear interpolation
    #weight = (target_density - rho0) / (rho1 - rho0)
    #interpolated_y = y0 + weight * (y1 - y0)
    y_target = y0 + (target_density - rho0) * (y1 - y0) / (rho1 - rho0)
    return y_target#interpolated_y
#%%

def find_before_below_target(arr, idx, target_density, target_closest):
    """
    Search backwards from idx to find the last element:
    - less than target_density
    - different from target_closest
    """
    for i in range(len(row[idx:])):
        if arr[idx+i]<target_density:
            return idx+i,arr[idx+i]
    
def find_before_above_target(arr, idx, target_density, target_closest):
    """
    Search backwards from idx to find the last element:
    - less than target_density
    - different from target_closest
    """
    for i in range(len(row[:idx]),0,-1):
        if arr[i]>target_density:
            return i,arr[i]

#%%
sim_configs = [
    {
        "path": "E:/PC_LULI/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_parl_21T_Foam_false_ideal/object5/",
        "label": "21 T par shock (ID 363325)"  # Parallel
    },
    {
        "path": "C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false/",
        "label": "No B-field (ID 362730)"
    },
    {
        "path": "C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_21T_Foam_false_ideal/RadMag_CH25Tin6_100mbar_Xe_B_perp_21T_Foam_false_ideal/",
        "label": "21 T ⊥ shock (ID 363314)"  # Perpendicular
    },
    {
        "path": "C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_21T_Foam_false_resistive/",
        "label": "21 T ⊥ Diff shock (ID 383113)"  # Perpendicular
    },
    {"path": 'C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false_Power_20Percent/',
     "label": "No B-field, 20 percent "
     },
    {"path": 'C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false_Power_40Percent/',
     "label": "No B-field, 40 percent "
     }, 
    {"path": 'C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false_Power_307,747Percent/',
     "label": "No B-field, 307,747 percent "
     }
    
]

#%%
plt.close('all')
# === FOLDER PATH ===
numberList = 2
MasterPath = sim_configs[numberList]['path']#'C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_21T_Foam_false_ideal/RadMag_CH25Tin6_100mbar_Xe_B_perp_21T_Foam_false_ideal/'#'C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false/'#'C:/Users/oportus.d/Documents/Flash_Diego/RadMag_CH25Tin6_100mbar_Xe_B_perp_0T_Foam_false_Power_307,747Percent/'
MasterPathSave = 'C:/Users/oportus.d/Documents/Flash_Diego/plot/'
#%%
field = 'brem'
target_density = 1.6e19  # 1/cm^3
if field == 'ne':
    units = 'cm**-3'
elif field != 'ne':
    units = 'eV'
elif field == 'brem':
    units = 'W/m**3'
if field!='ne':
    target_density = 'max'

window_um = 150
#%%
# === CONFIGURATION ===
K2eV = 1 / (Boltzmann / eV)
#%
species = {'shie': 'Sn','ablt': 'polystyrene','cham':'Xe'} #shie = pusher
#%%
# === DERIVED FIELDS FOR yt ===

#%%
json_folder = r'C:\Users\oportus.d\Documents\Flash_Diego\SOP2D_flash'

Z_spec = dict()
#for i, j in species.items():
    #Z_spec[i] = hedp.matdb(j)['Z']

for key, base in species.items():
    filename = os.path.join(json_folder, base + '.json')
    print(f"Looking for file: {filename}")
    
    if not os.path.isfile(filename):
        print(f"Warning: {filename} not found.")
        continue
    
    with open(filename, 'r') as f:
        data = json.load(f)
        Z_spec[key] = data['Z']
#%%    
def _Zcell(field, data):
    Zcell = 0
    for i in species.keys():
        Zcell += data[i] * Z_spec[i]
    return Zcell
yt.add_field('Zcell', function=_Zcell, units='dimensionless',sampling_type='cell')
#%%
def _Zion(field, data): return data['ye'] / data['sumy']
yt.add_field('Zion', function=_Zion, units='dimensionless', sampling_type='cell')

def _ne(field, data):
    amu = YTQuantity(1.66054e-24, "g")
    rho = data['dens'].to('g/cm**3')
    return (rho * data['ye'] / (amu)).in_units('1/cm**3')
yt.add_field('ne', function=_ne, units='1/cm**3', sampling_type='cell')
#%%
#%%
def ion_energy(zion,zbar):
    #return energy in J
    ei = 13.6 * (2/3)**(2/3) *e_ch*  zbar**(4/3) * (zion/zbar)**2 / (1-(zion/zbar))**(2/3)
    return ei
def neutral(ne,ni,temperature,ion_energy):
    #densities in cm**-3
    #temperature in K
    ne = 1e6 * ne #to pass m-3 SI
    ni = 1e6*ni
    u_plus = 1
    u = 2
    factor = 2 * ((2 * np.pi * m_e * kb * temperature) / h**2)**(3/2) * u_plus * np.exp(-ion_energy/(kb*temperature))/u
    n = ne*ni/factor 
    n = n*1e-6 #return neutral density in cm**-3
    return n

def photon_energy(lambd):
    #lambda in SI, return in J
    return h*c/lambd

def absoprtion(n,zion,zbar,temperature,lambd):
    #T in K, n in cm**-3, lambd in m
    #absoprtion in cm**-1
    energy_p = photon_energy(lambd)
    energy_i = ion_energy(zion,zbar)
    x1 = energy_i / (kb * temperature)
    x0 = energy_p / (kb * temperature)
    
    # Vectorized conditional
    
    factor = np.where(energy_i > energy_p,
                      np.exp(-(x1 - x0)),
                      2 * x1)
    
    # == calculate absorption coef in cm-1===
    abs_coeff_matrix = (0.96e-07 * n * (zion+1)**2 / temperature**2) * factor / x0**3
    return abs_coeff_matrix
def absoprtion_Vinci(n,zion,zbar,temperature,lambd):
    #T in K, n in cm**-3, lambd in m
    #absoprtion in cm**-1
    energy_p = photon_energy(lambd)
    energy_i = ion_energy(zion,zbar)
    x1 = energy_i / (kb * temperature)
    x0 = energy_p / (kb * temperature)
    
    # Vectorized conditional
    # == calculate absorption coef in cm-1===
    abs_coeff_matrix = (7.13e-16 * n * (zion+1)**2 / (temperature/K2eV)**2) * np.exp(-(x1 - x0)) / x0**3
    return abs_coeff_matrix

def absoprtion_Vinci2(n,zion,zbar,temperature,lambd):
    #T in K, n in cm**-3, lambd in m
    #absoprtion in cm**-1
    energy_p = photon_energy(lambd)
    energy_i = ion_energy(zion,zbar)
    x1 = energy_i / (kb * temperature)
    x0 = energy_p / (kb * temperature)
    
    # Vectorized conditional
    
    factor = np.where(energy_i > energy_p,
                      np.exp(-(x1 - x0)),
                      2 * x1)
    
    # == calculate absorption coef in cm-1===
    abs_coeff_matrix = (7.13e-16 * n * (zion+1)**2 / (temperature/K2eV)**2) * factor / x0**3
    return abs_coeff_matrix
def absoprtion2(n,zion,zbar,temperature,lambd):
    #T in K, n in cm**-3, lambd in m
    #absoprtion in cm**-1
    n = 1e6 * n #to pass to SI
    nu = c/lambd #frequency in SI, i.e Hz
    numerator = 64 * np.pi**4 * e_ch**10 * m_e * zion**4 * n
    denominator = 3 * np.sqrt(3) * h**6 * c * nu**3
    
    energy_p = photon_energy(lambd)
    energy_i = ion_energy(zion,zbar)
    x1 = energy_i / (kb * temperature)
    x0 = energy_p / (kb * temperature)
    
    term1 = np.exp(-x1) * (np.exp(x0)-1)/(2*x1)
    term2 = np.exp(-x1)/(2*x1)

    total = numerator * (term1 + term2)/denominator

    total = total*1e-2 #to pass from m-1 to cm-1
    
    # Vectorized conditional
    

    # == calculate absorption coef in m-1===
    #abs_coeff_matrix = (0.96e-07 * n * zion**2 / temperature**2) * factor / x0**3
    return total #abs_coeff_matrix

def absoprtion_silva(ne,ni,zion,zbar,temperature,lambd):
    nu = c/lambd
    ne = 1e6*ne
    ni = 1e6*ni
    factor = 0.75 * ((2 * np.pi) / (3*m_e * kb*temperature))**(1/2) * zion**2 * e_ch**6 * ni*ne / (h*c*m_e * nu**3)
    factor = 1e-2*factor #to pass from m-1 to cm-1
    return factor 
#%%
# === MAIN LOOP ===
file_list = sorted([
    f for f in glob(os.path.join(MasterPath, "lasslab_hdf5_plt_cnt_*"))
    if os.path.isfile(f) and "0000" not in os.path.basename(f)
])
#%%
sop_matrix = []
time_array = np.array([])
ti_matrix = []
for filename in file_list:#file_list:#[27]:#range(PLT_max_number):
    if os.path.exists(filename):
        lambd = 650*1e-9
        start_pix = 86 #to not include optical absoprtion commin from the rear side. We just care about the plasma radiation
        ds = yt.load(filename)
        time_array = np.append(time_array,ds.current_time.to('ns').value)
        slc =  yt.SlicePlot(ds, 'z', 'dens') #ds.covering_grid(level=0, left_edge=left_edge, dims=ds.domain_dimensions)
        data = slc.frb
        #%%
        # Get the 2D arrays directly
        te = data['tele'][start_pix:,:].value         # shape (M, N)
        zion = data['Zion'][start_pix:,:].value       # shape (M, N)
        zbar = data['Zcell'][start_pix:,:].value       # shape (M, N)
        energy_i = ion_energy(zion,zbar)#13.6 * (2/3)**(2/3) *e_ch*  zbar**(4/3) * (zion/zbar)**2 / (1-(zion/zbar))**(2/3) #* zion**2
        ne = data['ne'][start_pix:,:].value
        ni = data['ne'][start_pix:,:].value / zion        # shape (M, N)
        n = neutral(ne,ni,te,energy_i)
        if field == 'brem':
            abs_coeff_matrix = absoprtion(n,zion,zbar,te,lambd)
            Ipl =  (2 * h * c**2 / lambd**5) / (np.exp( h * c / (lambd*kb * te)) - 1)
            #x_vals_um_center = data['x'].to('um').value[Ipl.shape[0] // 2, :] 


        #%%
        x_vals_um_center = data['x'].to('um').value[data['tele'][start_pix:,:].shape[0] // 2, :] 
        dx_um = np.diff(x_vals_um_center) 
        dx_um_flat = dx_um.flatten()
        mean_dx_um = np.mean(dx_um_flat)
        std_dx_um = np.std(dx_um_flat)
        #print(f"Mean pixel spacing in x: {mean_dx_um:.3f} µm")
        #print(f"Standard deviation: {std_dx_um:.3f} µm")
        

        # Define window in microns
        
        half_window_um = window_um / 2  # = 75 μm
        
        # Extract a representative row (x position across columns)
        x_axis_um = x_vals_um_center # Shape: (Nx,)
        
        # Find indices where x is within [-75 μm, +75 μm]
        within_window = np.where((x_axis_um >= -half_window_um) & (x_axis_um <= half_window_um))[0]
        
        # Get the bounds
        i_min = within_window[0]
        i_max = within_window[-1]
        
        #print(f"x range: {x_axis_um[i_min]:.2f} µm to {x_axis_um[i_max]:.2f} µm")
        #print(f"Pixel index range: i_min = {i_min}, i_max = {i_max}")



        # == calculate absorption coef in m-1===
        #%% no we calculate the planck emission
        # blakc body emmision in IS units ==
#%%
        # Convert y to meters for calculations
        y_vals = data['y'].to('m').value[start_pix:, :]  # Shape: (Ny, Nx)
        if field == 'brem':
            I_vals = Ipl  # Shape: (Ny, Nx)
            k_vals = abs_coeff_matrix  # Shape: (Ny, Nx)

        else:
            I_vals = data[field][start_pix:,:].value 
        # Initialize outputs
        Ny, Nx = I_vals.shape
        I_new_matrix = np.zeros_like(I_vals)
        tau_matrix = np.zeros_like(I_vals)
        
        if field == 'brem':
            # Loop over columns (x-direction)
            for j in range(Nx):
                I_test = I_vals[:, j]
                y_test = y_vals[:, j]
                k_test = k_vals[:, j]
                tau_tot = 0
                for i in range(Ny):
                    if i == 0:
                        tau_matrix[i, j] = 0
                        I_new_matrix[i, j] = I_test[i]
                    else:
                        dy = abs(y_test[i] - y_test[i-1])
                        tau = dy * 0.5 * (k_test[i] + k_test[i-1])
                        tau_tot += tau
                        tau_matrix[i, j] = tau_tot
                        I_new_matrix[i, j] = I_test[i] * k_test[i] * dy * np.exp(-tau)
        else:
            I_new_matrix = data[field][start_pix:,:].value 
#%%
        

        
        I_window = I_new_matrix[:,i_min:i_max]
        if field == 'ne':
            ti_profile = data['tion'][start_pix:,:][:,i_min:i_max].value
            ti_profile_avg = np.mean(ti_profile, axis=1)
            ti_matrix.append(ti_profile_avg)


        I_profile_avg = np.mean(I_window, axis=1)  # shape: (Ny,)
        sop_matrix.append(I_profile_avg)
        
        if filename == file_list[-1]:
            y_array = data['y'].to('cm').value[start_pix:, (i_min + i_max)//2]
#%%
        
#%%
        # Plot original intensity (Ipl)
        if False:
            eps = 1e-30
            plt.figure()
            plt.imshow(Ipl + eps, aspect='auto', origin='lower')#, norm=LogNorm(vmin=eps, vmax=Ipl.max()))
            plt.colorbar(label='Original Intensity (log scale)')
            plt.title('Original Ipl (log scale)')
            plt.xlabel('X pixel')
            plt.ylabel('Y pixel')
    
            # Plot absorbed intensity (I_new_matrix)
            plt.figure()
            plt.imshow(I_new_matrix , aspect='auto', origin='lower')#, norm=LogNorm(vmin=eps, vmax=I_new_matrix.max()))
            plt.colorbar(label='Absorbed Intensity (log scale)')
            plt.title('I_new_matrix with Absorption (log scale)')
            plt.xlabel('X pixel')
            plt.ylabel('Y pixel')
                #%%
    
            sop_matrix.append(I_profile_avg)
            plt.figure()
            plt.imshow(I_window , aspect='auto', origin='lower')#, norm=LogNorm(vmin=eps, vmax=Ipl.max()))
            plt.colorbar(label='Original Intensity (log scale)')
            plt.title('Original Ipl (log scale)')
            plt.xlabel('X pixel')
            plt.ylabel('Y pixel')
            #%%
            plt.figure()
            plt.plot(data['y'].to('cm').value[start_pix:, (i_min + i_max)//2], I_new_matrix[:,(i_min + i_max)//2])
            plt.plot(data['y'].to('cm').value[start_pix:, (i_min + i_max)//2], I_profile_avg)

            #%%
        
#%%
sop_matrix = np.vstack(sop_matrix)
if field == 'tion':
    sop_matrix = sop_matrix/K2eV

if field == 'ne':
    ti_matrix = np.vstack(ti_matrix)

#%%
density_matrix2d = sop_matrix
i = 0
if field == 'ne':
    target_positions_updated = np.array([])
    for i, row in enumerate(density_matrix2d):
        print(i)
        #row = density_matrix2d[1]
        idx_ti = np.where(ti_matrix[i] == np.max(ti_matrix[i]))[0][0]
        # Check bounds
        target_closest = find_nearest(row[idx_ti:],target_density )
        idx_closest = np.where(row == find_nearest(row[idx_ti:],target_density ))[0][0]
        
        if target_closest<target_density:
            idx_above = find_before_above_target(row, idx_closest, target_density, target_closest)[0] #idx_closest-1
            idx_below = idx_closest
        elif target_closest>target_density:
            idx_above = idx_closest
            if idx_closest == len(row)-1:
                idx_below = len(row)-1
            else:
                idx_below = find_before_below_target(row, idx_closest, target_density, target_closest)[0] #idx_closest+1
        
        # Find bounding indices
       

        rho0, rho1 = row[idx_below], row[idx_above]
        y0, y1 = y_array[idx_below],y_array[idx_above]

        # Linear interpolation
        #weight = (target_density - rho0) / (rho1 - rho0)
        #interpolated_y = y0 + weight * (y1 - y0)
        y_target = y0 + (target_density - rho0) * (y1 - y0) / (rho1 - rho0)
        
        
        interpolated_y = y_target#interpolate_position_from_density(row, y_array, target_density)
        target_positions_updated = np.append(target_positions_updated,interpolated_y)
  
        
        if False:
            plt.figure()
            plt.plot(y_array,row,'.-')
            plt.plot(interpolated_y,target_density,'s')
            plt.plot()
            plt.yscale('log')
   
        #if int(time_array[i]) == 39:
        #    sdsd
        # === Plot for this time slice ===
        # === Plot for this time slice ===           
else:# field == 'tion':
    target_positions_updated = np.array([])
    for i, row in enumerate(density_matrix2d):
        print(i)
        #row = density_matrix2d[1]
        idx_ti = np.where(sop_matrix[i] == np.max(sop_matrix[i]))[0][0]

        target_positions_updated = np.append(target_positions_updated,y_array[idx_ti])
  
        
        if False:
            plt.figure()
            plt.plot(y_array,row,'.-')
            plt.plot(interpolated_y,target_density,'s')
            plt.plot()
            plt.yscale('log')
   
        #if int(time_array[i]) == 39:
        #    sdsd
        # === Plot for this time slice ===
        # === Plot for this time slice ===           
#%%
#%%

plt.figure(figsize=(10, 6))
extent = [y_array.min(), y_array.max(),time_array.max(), time_array.min()]
im = plt.imshow(sop_matrix,
           aspect='auto',
           extent=extent,
           cmap='jet',vmin = np.percentile(sop_matrix,0),vmax = np.percentile(sop_matrix,99.9))#,
           #norm=LogNorm(vmin=np.maximum(sop_matrix.min(), 1e-3), 
                        #vmax=sop_matrix.max())) 
            
if field == 'brem':
    field = 'Power Bremsstrahlung'
colorbarTitle = field+' ['+units+']'
plt.colorbar(im, label=colorbarTitle)  # ← Pass `im` to colorbar
plt.ylabel('Time [ns]')
plt.xlabel('Y [µm]')

plt.title(field + ' evolution along Y (avg over '+str(int(window_um))+ ' µm in X), ' + sim_configs[numberList]['label'])
plt.tight_layout()
if field== 'brem' or field == 'Power Bremsstrahlung':
    plt.savefig(MasterPathSave+'img/'+field+'_'+sim_configs[numberList]['label']+'_sop1d.png', dpi=300)
plt.show()            
#%%
plt.figure(figsize=(10, 6))
extent = [y_array.min(), y_array.max(),time_array.max(), time_array.min()]
im = plt.imshow(sop_matrix,
           aspect='auto',
           extent=extent,
           cmap='jet',
           norm=LogNorm(vmin=np.maximum(sop_matrix.min(), 1e-3), 
                        vmax=sop_matrix.max())) 
if field == 'ne' or field!='brem':
    if type(target_density) == str:
        density_str = target_density
    else:
        density_str = f"{target_density:.1e}"  # e.g., 5e10

    # Contour at 5e19
    #contour_level = 5e19
# Draw contour using the closest actual value
    mask = time_array > 2
    plt.plot(target_positions_updated[mask], time_array[mask],'.-',markersize = 9,color = 'black',label = field + ' = '+density_str+ ' '+units)
    plt.legend()
    filename = f"y_array_{field}_{sim_configs[numberList]['label']}_{density_str}_sop1d_unstack.npy"
    np.save(filename, target_positions_updated[mask])
    filename = f"time_array_{field}_{sim_configs[numberList]['label']}_{density_str}_sop1d_unstack.npy"
    np.save(filename,  time_array[mask])

if field == 'brem':
    field = 'Power Bremsstrahlung'
colorbarTitle = field+' ['+units+']'
plt.colorbar(im, label=colorbarTitle)  # ← Pass `im` to colorbar
plt.ylabel('Time [ns]')
plt.xlabel('Y [µm]')

plt.title(field + ' evolution along Y (avg over '+str(int(window_um))+ ' µm in X), ' + sim_configs[numberList]['label'])
plt.tight_layout()
if field!= 'brem':
    plt.savefig(MasterPathSave +'img/'+field+'_'+sim_configs[numberList]['label']+'_sop1d.png', dpi=300)
plt.show()
#%%
np.save(MasterPathSave +'matrix2d_'+field+'_'+sim_configs[numberList]['label']+'_sop1d_unstack.npy', sop_matrix)
np.save(MasterPathSave +'time_array_'+'_'+sim_configs[numberList]['label']+'_sop1d_unstack.npy', time_array)
np.save(MasterPathSave +'y_array_'+'_'+sim_configs[numberList]['label']+'_sop1d_unstack.npy', y_array)
#%%
# =============================================================================
# 
# =============================================================================
