# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:38:23 2024

@author: Tabitha
"""

import numpy as np
import qutip as qt
import scipy as sp
import math
import random
import simulation_2MQRM as t
import sys as s
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
import concurrent.futures
import pickle

N=20
D1=3
D2=3
ep1=0.2
ep2=0.2
wc=1
wa=1
geff_list_min = 0
geff_list_max = 2.6
geff_list_num = 2
geff_list = np.linspace(geff_list_min, geff_list_max, geff_list_num) 
additionscaling = [k**2/wc for k in geff_list]
lines = N*(D1+D2)

sets = 2
nsys = 100
normalised = True

def create_dataframe(nsys):
    outputDF = []
    svd2list = []
    svd1list = []
    for i in range(nsys):
        print(i/nsys)
        Cmatobj = t.CmatRandomAF(D1,D2,normalised)
        svd1list.append(Cmatobj.svdvals[0])
        svd2list.append(Cmatobj.svdvals[1])
        systems_temp_list = [t.DoubleMultilevel(N, D1, D2, Cmatobj, k, ep1, ep2 , wc, wa) for k in geff_list]
        systems_temp_list_energies = np.empty([geff_list_num], dtype = object)
        systems_temp_state_list = np.empty([geff_list_num], dtype = object)
        for k in range(geff_list_num):
            systems_temp_list[k].hamiltonian()
            systems_temp_list_energies[k], systems_temp_state_list[k] = np.array(systems_temp_list[k].H.eigenstates(sparse=True))
        energy_temp_list = np.empty([len(systems_temp_list_energies[0])],dtype=object) #energy levels specifically!!
        state_temp_list = np.empty([len(systems_temp_state_list[0])],dtype=object) #energy levels specifically!!
        for n in range(len(energy_temp_list)): #the length is the same as N*D because hamiltonian diagonalisation is the amount of energy levels
            energy_temp_list[n] = [systems_temp_list_energies[k][n] + additionscaling[k] for k in range(len(geff_list))]
            state_temp_list[n] = [systems_temp_state_list[k][n] for k in range(len(geff_list))]
    
        for n in range(len(state_temp_list)): #making actual state rho instead of vector
            for m in range(len(state_temp_list[0])):
                state_temp_list[n][m] = state_temp_list[n][m]*state_temp_list[n][m].dag()
    
        brightness_temp_list = np.empty([N*(D1+D2),len(geff_list)], dtype = np.float64)
    
        for n in range(len(brightness_temp_list)):
            for m in range(len(brightness_temp_list[0])):
                brightness_temp_list[n][m] = systems_temp_list[m].brightness_proportion(state_temp_list[n][m])
        
            brightness_temp_list[n] = [0 if math.isnan(x) else x for x in brightness_temp_list[n]] # forcing normalisation on lower value to be 0.
        
        linesDF = []
        nanDF = pd.DataFrame([np.nan],columns = ['x'])
        for k in range(lines): 
            dfsingleline = pd.DataFrame({'x': geff_list, 'y': energy_temp_list[k], 'weight':brightness_temp_list[k]})
            dfsinglelinewithnan = pd.concat([dfsingleline,nanDF])
            linesDF.append(dfsinglelinewithnan)
        outputDF.append(pd.concat(linesDF, ignore_index=True))
    return pd.concat(outputDF, ignore_index=True), svd1list, svd2list
        
def populate_dataframes_parallel(nsys, sets):
    bigset = []
    svd2list = []
    svd1list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks
        futures = [executor.submit(create_dataframe, nsys) for _ in range(sets)]
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            df, svdarray1, svdarray2 = future.result()
            bigset.append(df)
            svd1list.append(svdarray1)
            svd2list.append(svdarray2)
    
    return bigset, svd1list, svd2list

def main():
    bigdfset, svd1list, svd2list = populate_dataframes_parallel(nsys, sets)
    bigset = pd.concat(bigdfset, ignore_index=True)  
    svd1list = np.concatenate(svd1list)    
    svd2list = np.concatenate(svd2list)
    bigset.to_csv('multiprocessedenergies.csv', index=False)
    with open('svd1list', 'wb') as f:
        pickle.dump(svd1list, f)
    with open('svd2list', 'wb') as f:
        pickle.dump(svd2list, f)
    print('done :)')
    

if __name__ == '__main__':
    main()
