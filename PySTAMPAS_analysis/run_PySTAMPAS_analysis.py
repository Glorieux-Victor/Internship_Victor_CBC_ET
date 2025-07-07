import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('/home/victor-glorieux/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/src')
sys.path.append('/home/victor/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/src')
sys.path.append('/home/victor-glorieux/Internship_Victor_CBC_ET')
sys.path.append('/home/victor/Internship_Victor_CBC_ET')
from fonctions import extract_mchirp_tc_spectro
from get_data import read_MDC_data

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
true_i_closest = np.unique(i_closest - 1)
list_signals = injs.iloc[true_i_closest]
list_signals.index = np.arange(0, len(list_signals))

list_signals.to_csv('results/list_signals_SNR', index = False, sep = ' ')


# =================================== Mchirp and tc estimation ==============================================

# selected = [0]
# list_mchirp = []
# list_tc = []
# for i in selected :
#     list_signals_sel = list_signals.iloc[i]
#     data = read_MDC_data(list_signals_sel['t0'], list_signals_sel['tf'])
#     dict = extract_mchirp_tc_spectro(data,'E1',q_lim = 100,show_fit=True,save_fig=True)

#     list_mchirp.append(dict['mchirp'])
#     list_tc.append(dict['tc'])

# list_signals['mchirp'] = list_mchirp
# list_signals['tc'] = list_tc




