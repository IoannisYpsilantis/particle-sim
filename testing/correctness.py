"""
This file is responsible for providing a way to compare the output particle positions of simulations.
"""

import pandas as pd
import numpy as np
import os
import sys



def read_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)
    
    with open(file_path, "r") as f:
        header = f.readline().strip().split(" ")
        header = [h.split(":") for h in header]
        dev = file_path.split("_")[-2]
        header.append(["device", dev])
        header = pd.DataFrame(header)
        header.columns = ["name", "value"]
        header.set_index("name", inplace=True)
        
            
    data = pd.read_csv(file_path, sep=" ", skiprows=1, header=None, )
    return header.T, data



"""
This functions expects the sim1 and sim2 to have a shape of:
(num_particles, 4)
"""
def compare_positions(num_particles, sim1, sim2, tol = 1e-3):
    
    # Check if the positions are the same
    counter_examples = []
    for i in range(num_particles):
        for j in range(3):
            if np.abs(sim1[i][j] - sim2[i][j]) > tol:
                counter_examples.append((i, j))
    if len(counter_examples) > 0:
        return False, counter_examples
    return True, counter_examples


def get_timing(id, directory):
    files = os.listdir(directory)
    numParticles = []
    time = []
    finalPos = {}
    for file in files:
        if file.split("_")[-1] == str(id) + ".txt":
            header, data = read_file(os.path.join(directory, file))
            numParticles.append(int(header["particles"]))
            time.append(float(header["timing"])/float(header["iterations"]))
            finalPos[int(header["particles"])] = data
    return pd.DataFrame({"particles": numParticles, "timing": time}).sort_values(by=["particles"]), finalPos
            