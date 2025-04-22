import re 
import numpy as np

class FileParsingError(Exception):
    """Custom exception for header parsing errors."""
    pass

def readMarsGRAM(file,vars):

    header = readASCIIheader(file)
    temp = np.loadtxt(file, skiprows=header['skiprows'],usecols=vars)

    data = {}

    nlons = header['nlons']
    nlats = header['nlats']
    nalts = header['nalts']
    for i in range(len(vars)):
        values = temp[:,i]
        data[vars[i]]=values.reshape((nlons,nlats,nalts))
  
    return data 

def readASCIIheader(file):
    nlats = nlons = nalts = None
    linecount = 0
    started = False
    vars = None 
    with open(file,'r') as f:

        for linecount, line in enumerate(f,start=1):
            if not line.strip():
                continue

            lat_match = re.search(r'Number of Latitude points:\s+(\d+)', line)
            lon_match = re.search(r'Number of Longitude points:\s+(\d+)', line)
            alt_match = re.search(r'Number of Altitude points:\s+(\d+)', line)
            if re.search(r'\bLongitude\b', line) and re.search(r'\bLatitude\b', line) and \
                re.search(r'\bAltitude\b', line):
                vars = re.split(r'\s{2,}', line.strip())

            if lat_match:
                nlats = int(lat_match.group(1))
            if lon_match:
                nlons = int(lon_match.group(1))
            if alt_match:
                nalts = int(alt_match.group(1))

            start_match = re.search(r'#START', line)
            if start_match:
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
    

