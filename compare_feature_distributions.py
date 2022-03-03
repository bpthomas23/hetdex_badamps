import numpy as np
#from build_feature_classifier import BinaryClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from tensorflow.keras.optimizers import Adam

import pandas as pd

print('~~~loading data~~~')
amp_stats2 = np.load('amp_stats_hdr2.npy')
amp_stats3 = np.load('amp_stats_hdr3.npy')

print('~~~convert to pandas~~~')
amp_stats2_df = pd.DataFrame(amp_stats2[1:], columns=amp_stats2[0])
amp_stats3_df = pd.DataFrame(amp_stats3[1:], columns=amp_stats3[0])

id2_df = amp_stats2_df.iloc[:,:3]
flag2_df = amp_stats2_df.iloc[:,-3:]
features2_df = amp_stats2_df.iloc[:,3:-3]

idf3_df = amp_stats3_df.iloc[:,:3]
features3_df = amp_stats3_df.iloc[:,3:]

#there are a few rows with MaskFraction ='""', drop them
mask2 = features2_df.MaskFraction == '""'
features2_df = features2_df[~mask2]

print(len(np.where(mask2)[0]), ' rows dropped due to missing values')

#also drop date columns
features2_df = features2_df.drop(['date', 'Date'], axis=1)
features3_df = features3_df.drop(['Date'], axis=1)

features2_df = features2_df.astype('float')
features3_df = features3_df.astype('float')

columns2 = features2_df.columns
columns3 = features3_df.columns

columndiff = np.array([column for column in columns2 if column not in columns3])
print('columns ', columndiff, ' are present in hdr2.1 but not in hdr3')

"""
pd.plotting.scatter_matrix(features_df.sample(n=1000), alpha=0.1, hist_kwds={'bins':100})

plt.savefig('scatter_matrix.pdf')
"""

for feature_name in features2_df.columns:
	dist = features2_df[feature_name]
	plt.figure()

	if feature_name not in columndiff:
		if ((feature_name == 'background') | (feature_name=='norm')):
			bins=15
		else:
			bins=100
		dist3 = features3_df[feature_name]
		fulldist = np.append(dist,dist3)
		mean = fulldist.mean()
		std = fulldist.std()

		plt.hist(dist3, bins=bins,range=(mean-3*std, mean+3*std), alpha=0.5,label='hdr3')
	plt.hist(dist, bins=100, range=(mean-3*std, mean+3*std), alpha=0.5,label='hdr2.1')
	plt.xlabel(feature_name)
#	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.savefig('eda_plots/'+feature_name+'.pdf')
	plt.close()


