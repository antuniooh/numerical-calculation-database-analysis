import pandas as pd
import numpy as np

def readData():
    data = pd.read_csv('dadoscartola_Atualizado.csv', header=(0))
    ylabel = data.columns[-1]

    data = data.to_numpy()
    nrow,ncol = data.shape
    y = data[:,-1]
    X = data[:,0:ncol-1]