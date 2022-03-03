import numpy as np
import pandas as pd

amp_stats = np.load('amp_stats_hdr3.npy')
amp_stats = pd.DataFrame(amp_stats[1:], columns=amp_stats[0])

features = amp_stats.iloc[:,3:]

mask = features.norm == '""'
features = features[~mask]
features = features.astype('float')
print(np.sum(mask), ' rows dropped due to missing values')
features=features.drop('Date',axis=1)

mean = features.mean(axis=0)
std = features.std(axis=0)

stats = pd.concat([mean,std], axis=1)
stats.columns = ['mean','std']

header = stats.columns
rownames = stats.index

savarr = np.row_stack((header, stats.to_numpy()))
savarr = np.column_stack((rownames.insert(0, ' '), savarr))



np.save('hdr3_normalisation.npy', savarr)
