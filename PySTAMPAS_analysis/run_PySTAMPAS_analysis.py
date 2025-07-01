import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data reading

triggers_noise = np.load('inputs/triggers_background_Gaussian_noise.npy')
triggers_MDC = np.load('inputs/triggers_background_MDC.npy')

triggers = {'t_start' : triggers_MDC['start1'], 't_end' : triggers_MDC['start1'] +  triggers_MDC['duration'], 'duration' : triggers_MDC['duration'], 'fmin' : triggers_MDC['fmin'], 'fmax' : triggers_MDC['fmax']}
dataframe = pd.DataFrame(triggers)
dataframe = dataframe.drop_duplicates(subset=['t_start'])
dataframe['type'] = 0

for i in dataframe.index.values :
    if dataframe.loc[i,'fmin'] < 10 and dataframe.loc[i,'fmax'] < 200 :
        dataframe.loc[i,'type'] = 3
    else :
        dataframe.loc[i,'type'] = 2

print(dataframe['duration'].max())