#!/usr/bin/env python

from glob import glob
from datetime import datetime
from datetime import timedelta
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from pylab import cm
import re 
import pandas as pd 
import os

#-----------------------------------------------------------------------------

class FileParsingError(Exception):
    """Custom exception for header parsing errors."""
    pass

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def file_time(fname):
    m = re.search(r'_t(\d{6}_\d{6}).', os.path.basename(fname))
    if m:
        return datetime.strptime(m.group(1), '%y%m%d_%H%M%S')
    return None

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def read_gitm_header(file):

    if (len(file) == 0):
        filelist = glob('./3DALL*.bin')

        if (len(filelist) == 0):
            print("No 3DALL files found. Checking for 1DALL.")
            filelist = glob('./1DALL*.bin')
            if (len(filelist) == 0):
                print("No 1DALL files found. Stopping.")
                exit()
            file = filelist[0]

    else:
        filelist = glob(file[0])
        file = filelist[0]

    
    header = {}
    header["nFiles"] = len(filelist)
    header["version"] = 0
    header["nLons"] = 0
    header["nLats"] = 0
    header["nAlts"] = 0
    header["nVars"] = 0
    header["vars"] = []
    header["time"] = []
    header["filename"] = []

    header["filename"].append(file)

    f=open(file, 'rb')

    # This is all reading header stuff:

    endChar='>'
    rawRecLen=f.read(4)
    recLen=(unpack(endChar+'l',rawRecLen))[0]
    if (recLen>10000)or(recLen<0):
        # Ridiculous record length implies wrong endian.
        endChar='<'
        recLen=(unpack(endChar+'l',rawRecLen))[0]

    # Read version; read fortran footer+header.
    header["version"] = unpack(endChar+'d',f.read(recLen))[0]

    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Read grid size information.
    (header["nLons"],header["nLats"],header["nAlts"]) = unpack(endChar+'lll',f.read(recLen))
    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Read number of variables.
    header["nVars"]=unpack(endChar+'l',f.read(recLen))[0]
    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Collect variable names.
    for i in range(header["nVars"]):
        v = unpack(endChar+'%is'%(recLen),f.read(recLen))[0]
        header["vars"].append(v.decode('utf-8').replace(" ",""))
        (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Extract time.
    (yy,mm,dd,hh,mn,ss,ms)=unpack(endChar+'lllllll',f.read(recLen))
    header["time"].append(datetime(yy,mm,dd,hh,mn,ss,ms*1000))
    # print(header["time"][-1])

    f.close()

    return header

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
def read_ascii_header(file):
    nlats = nlons = nalts = None
    linecount = 0
    started = False
    vars = None 
    with open(file,'r') as f:

        for linecount, line in enumerate(f,start=1):
            if not line.strip():
                continue

            if "Number of Latitude points:" in line:
                nlats = int(line.split(":")[1].strip())
            if "Number of Longitude points:" in line:
                nlons = int(line.split(":")[1].strip())
            if "Number of Altitude points:" in line:
                nalts = int(line.split(":")[1].strip())

            if 'Longitude' in line and 'Latitude' in line and \
                'Altitude'in line:
                vars = re.split(r'\s{2,}', line.strip())

            if '#START' in line:
                started = True 
                break 

    if not started:
        raise FileParsingError("No '#START' line found in file.")
    if not (nlats and nlons and nalts):
        raise FileParsingError("Missing grid size information in file header")

    if vars is None:
        raise FileParsingError("Missing variable names in the header")

    return {'nlons':nlons,
        'nlats':nlats,
        'nalts':nalts,
        'vars':vars,
        'skiprows': linecount,
        } 

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

def read_gitm_one_file(file_to_read, vars_to_read=-1):

    # print("Reading file : "+file_to_read)

    data = {}
    data["version"] = 0
    data["nLons"] = 0
    data["nLats"] = 0
    data["nAlts"] = 0
    data["nVars"] = 0
    data["time"] = 0
    data["vars"] = []

    f=open(file_to_read, 'rb')

    # This is all reading header stuff:

    endChar='>'
    rawRecLen=f.read(4)
    recLen=(unpack(endChar+'l',rawRecLen))[0]
    if (recLen>10000)or(recLen<0):
        # Ridiculous record length implies wrong endian.
        endChar='<'
        recLen=(unpack(endChar+'l',rawRecLen))[0]

    # Read version; read fortran footer+data.
    data["version"] = unpack(endChar+'d',f.read(recLen))[0]

    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Read grid size information.
    (data["nLons"],data["nLats"],data["nAlts"]) = unpack(endChar+'lll',f.read(recLen))
    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Read number of variables.
    data["nVars"]=unpack(endChar+'l',f.read(recLen))[0]
    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    if (vars_to_read[0] == -1):
        vars_to_read = np.arange[nVars]

    # Collect variable names.
    for i in range(data["nVars"]):
        data["vars"].append(unpack(endChar+'%is'%(recLen),f.read(recLen))[0])
        (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Extract time.
    (yy,mm,dd,hh,mn,ss,ms)=unpack(endChar+'lllllll',f.read(recLen))
    data["time"] = datetime(yy,mm,dd,hh,mn,ss,ms*1000)
    #print(data["time"])

    # Header is this length:
    # Version + start/stop byte
    # nLons, nLats, nAlts + start/stop byte
    # nVars + start/stop byte
    # Variable Names + start/stop byte
    # time + start/stop byte

    iHeaderLength = 8 + 4+4 + 3*4 + 4+4 + 4 + 4+4 + data["nVars"]*40 + data["nVars"]*(4+4) + 7*4 + 4+4

    nTotal = data["nLons"]*data["nLats"]*data["nAlts"]
    iDataLength = nTotal*8 + 4+4

    for iVar in vars_to_read:
        f.seek(iHeaderLength+iVar*iDataLength)
        s=unpack(endChar+'l',f.read(4))[0]
        data[iVar] = np.array(unpack(endChar+'%id'%(nTotal),f.read(s)))
        data[iVar] = data[iVar].reshape(
            (data["nLons"],data["nLats"],data["nAlts"]),order="F")

    f.close()

    return data



def read_gitm_ascii_onefile(file, vars_to_read=-1):
    """Read gitm ASCII file."""

    data = {}
    data["version"] = 0  # ASCII may not have version info
    data["nLons"] = 0
    data["nLats"] = 0
    data["nAlts"] = 0
    data["nVars"] = 0
    data["time"] = None
    data["vars"] = []

    header = read_ascii_header(file)
    data["nLons"] = header["nlons"]
    data["nLats"] = header["nlats"]
    data["nAlts"] = header["nalts"]
    data["vars"] = header["vars"]
    data["nVars"] = len(header["vars"])
    
    # If vars_to_read not passed, read all variables
    if vars_to_read[0] == -1:
        vars_to_read = list(range(data["nVars"]))
    
    temp_df = pd.read_csv(file,
                delim_whitespace=True,
                skiprows=header['skiprows'],
                header=None,
                usecols=vars_to_read)
    
    data_array = temp_df.values
    reshaped = data_array.reshape((header['nlons'], header['nlats'], header['nalts'], -1))
    vars_to_read.sort() #Helps ensure var mapping doesn't get screwed up

    for i, var_index in enumerate(vars_to_read):
        data[var_index] = reshaped[:, :, :, i]
    return data

def extract_year(filename,pattern):
    match = re.search(pattern, filename)
    
    if match:
        year = int(match.group(2)[:2])  # Extract the first two digits as the year

        if year < 50:
            # For years less than 50, assume 20xx
            year += 2000
        else:
            # For years greater than or equal to 50, assume 19xx
            year += 1900
        return year

    else:
        print('Error')
        return None


def extract_timestamp(filename,pattern):
    match = re.search(pattern, filename)
    if match:
        timestamp = datetime.strptime(str(extract_year(filename,pattern))+match.group(2)[2:]+ match.group(3),\
            '%Y%m%d%H%M%S')
        return timestamp
    else:
        print("Error")
        return None

def parse_filename(filepath):
    #parse filename for sorting
    filename = os.path.basename(filepath)
    base = filename.split('.')[0]  # remove extension
    parts = base.split('_')
    
    # Handle formats
    if parts[0].startswith('GITM'):
        # Format: GITM_YYMMDD_HHMMSS
        yymmdd = parts[1]
        hhmmss = parts[2]
    elif parts[0].startswith('3D'):
        # Format: 3D???_tYYMMDD_HHMMSS
        if not parts[1].startswith('t'):
            raise ValueError(f"Expected 't' in second part of filename {filename}")
        yymmdd = parts[1][1:]  # Skip the 't'
        hhmmss = parts[2]
    else:
        raise ValueError(f"Unrecognized filename format: {filename}")

    # Parse components
    yy = int(yymmdd[0:2])
    mo = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    hh = int(hhmmss[0:2])
    mi = int(hhmmss[2:4])
    ss = int(hhmmss[4:6])

    # Expand the year
    year = 1900 + yy if yy >= 90 else 2000 + yy

    # Return a datetime object
    return datetime(year, mo, dd, hh, mi, ss)

def calculate_sza(lat, lon, date_time):
    '''Return sza given position (degrees) and datetime object'''

    lat_rad = np.radians(lat)

    # Constants
    days_in_year = 365.25

    day_of_year = date_time.timetuple().tm_yday

    # Calculate solar declination (δ)
    declination = 23.44 * np.cos(np.radians((360 / days_in_year) * (day_of_year + 10)))
    declination_rad = np.radians(declination)

    # Calculate time in hours and solar hour angle (H)
    time_of_day = date_time.hour + date_time.minute / 60 + date_time.second / 3600
    solar_noon = (12.0 - lon) / 15.0  # Adjust solar noon for the location

    hour_angle = (time_of_day - solar_noon) * 15  # 15 degrees per hour
    hour_angle_rad = np.radians(hour_angle)

    # Calculate the cosine of the solar zenith angle (SZA)
    cos_sza = (np.sin(lat_rad) * np.sin(declination_rad) +
               np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))

    # Ensure cos_sza is within valid range [-1, 1]
    # cos_sza = min(1, max(cos_sza, -1))

    # Calculate the solar zenith angle in degrees
    sza = np.degrees(np.arccos(cos_sza))

    return sza
#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

AMU = 1.66054e-27

mC = 12.011
mO = 15.9994
mN = 14.00674
mAr = 39.948

masses = {"[C]":mC,
"[O]":mO,
"[O!D2!N]":2*mO,
"[CO]":mC + mO,
"[N!D2!N]":2*mN,
"[Ar]":mAr,
"[NO]":mN + mO,
"[CO!D2!N]":mC + 2*mO
}

name_dict = {"Altitude":"Altitude",
                     "Ar Mixing Ratio":"Argon Mixing Ratio",
                     "Ar":"Ar Mass Density",
                     "CH4 Mixing Ratio":"Methane Mixing Ratio",
                     "Conduction":"Conduction", "EuvHeating":"EUV Heating",
                     "H":"H Mass Density", "H!U+!N":"H$^+$ Mass Density",
                     "H2 Mixing Ratio":"H$_2$ Mixing Ratio",
                     "HCN Mixing Ratio":"Hydrogen Cyanide Mixing Ratio",
                     "He":"He Mass Density", "He!U+!N":"He$^+$ Mass Density",
                     "Heating Efficiency":"Heating Efficiency",
                     "Heat Balance Total":"Heat Balance Total",
                     "Latitude":"Latitude", "Longitude":"Longitude",
                     "[N!D2!N]":"[N$_2$]",
                     "[N!D2!U+!N]":"[N$_2$$^+$]",
                     "[N!U+!N]":"[N$^+$]",
                     "[N(!U2!ND)]":"[N($^2$D)]",
                     "[N(!U2!NP)]":"[N($^2$P)]",
                     "[N(!U4!NS)]":"[N($^4$S)]",
                     "N2 Mixing Ratio":"N$_2$ Mixing Ratio",
                     "[NO]":"[NO]", "[NO!U+!N]":"[NO$^+$]",
                     "[O!D2!N]":"[O$_2$] ",
                     "[O(!U1!ND)]":"[O($^1$D)] ",
                     "[O!D2!U+!N]":"[O$_2$$^+$]",
                     "[O(!U2!ND)!]":"[O($^2$D)] ",
                     "[O(!U2!ND)!U+!N]":"[O($^2$D)] ",
                     "[O(!U2!NP)!U+!N]":"[O($^2$P)$^+$] ",
                     "[O(!U2!NP)!U+!N]":"[O($^2$P)] ",
                     "[O(!U3!NP)]":"[O($^3$P)] ",
                     "[O_4SP_!U+!N]":"[O($^4$SP)$^+$] ",
                     "RadCooling":"Radiative Cooling", "Rho":"Rho",
                     "Temperature":"T$_n$", "V!Di!N (east)":"v$_{east}$",
                     "V!Di!N(north)":"v$_{north}$", "V!Di!N(up)":"v$_{up}$",
                     "V!Dn!N(east)":"u$_{east}$",
                     "V!Dn!N(north)":"u$_{north}$", "V!Dn!N(up)":"u$_{up}$",
                     "V!Dn!N(up,N!D2!N              )":"u$_{Up, N_2}$",
                     "V!Dn!N(up,N(!U4!NS)           )":"u$_{Up, N(^4S)}$",
                     "V!Dn!N(up,NO                  )":"u$_{Up, NO}$",
                     "V!Dn!N(up,O!D2!N              )":"u$_{Up, O_2}$",
                     "V!Dn!N(up,O(!U3!NP)           )":"u$_{Up, O(^3P)}$",
                     "e-":"[e-]",
                     "Electron_Average_Energy":"Electron Average Energy",
                     "eTemperature":"T$_e$", "iTemperature":"T$_i$",
                     "Solar Zenith Angle":"Solar Zenith Angle",
                     "Vertical TEC":"VTEC", "CO!D2!N":"CO$_2$ Mass Density",
                     "DivJu FL":"DivJu FL", "DivJuAlt":"DivJuAlt",
                     "Electron_Energy_Flux":"Electron Energy Flux",
                     "FL Length":"Field Line Length",
                     "Pedersen FL Conductance":r"$\sigma_P$",
                     "Pedersen Conductance":r"$\Sigma_P$",
                     "Hall FL Conductance":r"$\sigma_H$",
                     "Potential":"Potential", "Hall Conductance":r"$\Sigma_H$",
                     "Je2":"Region 2 Current", "Je1":"Region 1 Current",
                     "Ed1":"Ed1", "Ed2":"Ed2", "LT":"Solar Local Time",
                     "E.F. Vertical":"Vertical Electric Field",
                     "E.F. East":"Eastward Electric Field",
                     "E.F. North":"Northward Electric Field",
                     "E.F. Magnitude":"Electric Field Magnitude",
                     "B.F. Vertical":"Vertical Magnetic Field",
                     "B.F. East":"Eastward Magnetic Field",
                     "B.F. North":"Northward Magnetic Field",
                     "B.F. Magnitude":"Magnetic Field Magnitude",
                     "Magnetic Latitude":"Magnetic Latitude",
                     "Magnetic Longitude":"Magnetic Longitude",
                     "dLat":"Latitude", "dLon":"Longitude", "Gravity":"g",
                     "PressGrad (east)":r"$\nabla_{east}$ (P$_i$ + P$_e$)",
                     "PressGrad (north)":r"$\nabla_{north}$ (P$_i$ + P$_e$)",
                     "PressGrad (up)":r"$\nabla_{up}$ (P$_i$ + P$_e$)",
                     "IN Collision Freq":r"$\nu_{in}$",
                     "Chemical Heating":"Chemical Heating Rate",
                     "Total Abs EUV":"Total Absolute EUV",
                     "O Cooling":"O Cooling", "Joule Heating":"Joule Heating",
                     "Auroral Heating":"Auroral Heating",
                     "Photoelectron Heating":"Photoelectron Heating",
                     "Eddy Conduction":"Eddy Conduction",
                     "Eddy Adiabatic Conduction":"Adiabatic Eddy Conduction",
                     "NO Cooling":"NO Cooling",
                     "Molecular Conduction":"Molecular Conduction",
                     "[CO!D2!N]":"[CO$_2$]",
                     "[O]":"[O]",
                     "[O!D2!N]":"[O$_2$]",
                     "[e-]":"[e-]",
                     "[CO]":"[CO]",
                     "[N!D2!N]":"[N$_2$]",
                     "[Ar]":"[Ar]",
                     "[NO]":"[NO]",
                     "[O!U+!N]":"[O$^+$]",
                     "[O!D2!U+!N]":"[O$_2^+$]",
                     "[CO!D2!U+!N]":"[CO$_2^+$]",
                     "[He]":"[He]",
                     "[N]":"[N]",
                     "[H]":"[H]",
                     "[N(2D)]":"[N(2D)]",
                     "V!Dn!N(up,CO!D2!N)":"V$_$n(up,CO$_2$)",
                     "[CO!U+!N]":"[CO$^+$]",
                     "[C!U+!N]":"[C$^+$]",
                     "[C]":"[C]",
                     }


def clean_varname(varname, netcdf_safe=False):

    cleanvar = (varname.strip()
                             .replace('$', '')
                             .replace('{', '')
                             .replace('}', '')
                             .replace('/', '')
                             .replace('!D','_')
                             .replace('!N','')
                             .replace('!U','')
                             .replace(" ","")
                             .replace("^","")
                             .replace('_',"")
    )

    if netcdf_safe:
        # Remove brackets
        cleanvar = cleanvar.replace('[','').replace(']','')

        # Remove any remaining illegal characters
        cleanvar = re.sub(r'[^A-Za-z0-9_]', '', cleanvar)

        # Ensure it starts with a letter
        if not cleanvar or not cleanvar[0].isalpha():
            cleanvar = "var_" + cleanvar

    return cleanvar

def autoscale_axis(ax, axis='y', digits=1, base_label=None,units=None, fontsize=10, italic=False,fixed_digits=None):
    """
    Automatically rescales axis tick labels and adds a multiplier like ×10ⁿ.

    Parameters:
    - ax: matplotlib Axes object
    - axis: 'x' or 'y'
    - digits: number of decimal digits to show on ticks
    - offset: (x, y) tuple in Axes coords for multiplier label
    - fontsize: size of multiplier text
    - italic: if True, uses italic style
    """
    assert axis in ['x', 'y'], "axis must be 'x' or 'y'"

    # Get current tick values
    ticks = ax.get_yticks() if axis == 'y' else ax.get_xticks()
    ticks = ticks[np.isfinite(ticks)]
    if len(ticks) == 0:
        return  # avoid errors with empty ticks

    # Determine order of magnitude
    max_val = max(abs(ticks.min()), abs(ticks.max()))
    if max_val == 0:
        exponent = 0
    else:
        exponent = int(np.floor(np.log10(max_val)))

    # No need to scale if exponent is close to 0
    if abs(exponent) < 4:
        return  # skip rescaling if not worth it

    scale = 10 ** exponent

    scaled_ticks = ticks / scale
    if fixed_digits is not None:
        digits = fixed_digits
    else:
        max_scaled = max(abs(scaled_ticks.min()), abs(scaled_ticks.max()))
        if max_scaled >= 100:
            digits = 0
        elif max_scaled >= 10:
            digits = 1
        elif max_scaled >= 1:
            digits = 2
        elif max_scaled >= 0.1:
            digits = 3
        else:
            digits = 4

   # Apply tick formatting
    formatted_ticks = [f"{val:.{digits}f}" for val in scaled_ticks]
    if axis == 'y':
        ax.ticklabel_format(axis='y', style='plain')
        ax.set_yticks(ticks)  # Freeze tick positions
        ax.set_yticklabels(formatted_ticks)
    else:
        ax.ticklabel_format(axis='x', style='plain')
        ax.set_xticks(ticks)  # Freeze tick positions
        ax.set_xticklabels(formatted_ticks)

    # Build new axis label
    exponent_str = rf"$\times 10^{{{exponent}}}$"
    style = 'italic' if italic else 'normal'

    if base_label is None:
        base_label = ax.get_ylabel() if axis == 'y' else ax.get_xlabel()

    if units:
        full_label = rf"{base_label} ({exponent_str} {units})"
    else:
        full_label = rf"{base_label} ({exponent_str})"

    # Set the new axis label
    if axis == 'y':
        ax.set_ylabel(full_label, fontsize=fontsize, style=style)
    else:
        ax.set_xlabel(full_label, fontsize=fontsize, style=style)

    return ax

marsreactions = {
  1:'N2 -> N2+',
  2:'CO2 -> CO2+',
  7:'O2 -> O2+',
  9:'O -> O+',

}