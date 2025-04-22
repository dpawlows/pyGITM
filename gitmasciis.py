import re 
import pandas as pd 
import concurrent.futures

class FileParsingError(Exception):
    """Custom exception for header parsing errors."""
    pass

def readMarsGRAM(file,vars):
    try: 
        header = readASCIIheader(file)

        data = {}
        temp_df = pd.read_csv(file,
                    delim_whitespace=True,  # matches loadtxt's default behavior
                    skiprows=header['skiprows'],
                    header=None,            # since your file has no header row for data
                    usecols=vars)

        for i, var in enumerate(vars):
                values = temp_df.iloc[:,i].values
                data[var] = values.reshape((header['nlons'], header['nlats'], header['nalts']))
        
        del temp_df

        return data 
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def process_batch(files, vars,max_workers=None):
    """Function to process a batch of files in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda file: readMarsGRAM(file,vars),files
                ))
    
    return results

def readASCIIheader(file):
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

            #if lat_match:
            #    nlats = int(lat_match.group(1))
            #if lon_match:
            #    nlons = int(lon_match.group(1))
            #if alt_match:
            #    nalts = int(alt_match.group(1))

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
    

