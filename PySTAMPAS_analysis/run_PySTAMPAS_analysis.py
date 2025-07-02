import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data reading

triggers_noise = np.load('inputs/triggers_background_Gaussian_noise.npy')
triggers_MDC = np.load('inputs/triggers_background_MDC.npy')

triggers = {'t_start' : triggers_MDC['start1'], 't_end' : triggers_MDC['start1'] +  triggers_MDC['duration'], 'duration' : triggers_MDC['duration'], 'fmin' : triggers_MDC['fmin'], 'fmax' : triggers_MDC['fmax'], 'p_lambda' : triggers_MDC['p_lambda']}
dataframe = pd.DataFrame(triggers)
dataframe = dataframe.drop_duplicates(subset=['t_start'])
dataframe['type'] = 0

for i in dataframe.index.values :
    if dataframe.loc[i,'fmin'] < 10 and dataframe.loc[i,'fmax'] < 200 :
        dataframe.loc[i,'type'] = 3
    else :
        dataframe.loc[i,'type'] = 2

injs = pd.read_csv('/home/victor/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/data/loudest_BBH/list_mdc1_v2.txt', sep = ' ', usecols = ['t0', 'tc', 'tf', 'snr', 'type'])
min_SNR_filter = 15
injs = injs[injs['snr'] > min_SNR_filter]
injs = injs.sort_values(by = 'snr', ascending = True)
injs.index = np.arange(0, len(injs))
tc = injs['tc']


def find_closest_tc(tc, t_end):
    dist = np.abs(tc - t_end) #make the diff for all values of tc with one t_end
    i_min = np.argmin(dist) #return the index of the minimum dist
    return i_min

pl = dataframe['p_lambda']
true_pl = np.zeros_like(pl)

i_closest = np.zeros(len(dataframe['t_end']), dtype=int)

i=0
for t_end in dataframe['t_end']:
    i_closest[i] = find_closest_tc(tc, t_end)
    i += 1

print('Total number of signals :', len(i_closest))
print('Number of forgotten signals :', len(i_closest) - len(np.unique(i_closest)))
list_signals = injs.iloc(i_closest)

print(list_signals)
