import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

print('~~~loading data~~~')
amp_stats2 = np.load('amp_stats_hdr2_cutflag.npy', allow_pickle=True)

print('~~~convert to pandas~~~')
amp_stats2_df = pd.DataFrame(amp_stats2[1:], columns=amp_stats2[0])

idcols = ['shotid', 'expnum', 'multiframe']
featurecols = ['norm','N_cont','background','sky_sub_rms','wave0','wave1','nfib_bad',
		'median_sky_rms','stdev_sky_rms','sky_sub_rms_rel','im_median',
		'MaskFraction']
flagcols = ['flag_manual','flag','ft_flag','cut_flag']


id_df = amp_stats2_df[idcols]
features_df = amp_stats2_df[featurecols]
flag_df = amp_stats2_df[flagcols]

##bad maskfraction rows already dropped from this dataset
features_df = features_df.astype('float')
#features_df.dropna(inplace=True)
mask = ~np.isnan(features_df.MaskFraction.to_numpy())

features_df = features_df[mask]
flag_df = flag_df[mask]
id_df = id_df[mask]

labels = flag_df['flag'].to_numpy(dtype='int')

#keep only as much good data as there is bad data, such that the classes have a 50/50 split
np.random.seed(42)
goodinds = np.where(labels == 1)[0]
badinds = np.where(labels == 0)[0]
newgoodinds = np.random.choice(goodinds, size=len(badinds), replace=False)
inds = np.append(badinds,newgoodinds)
np.random.shuffle(inds)

features_df = features_df.iloc[inds]
id_df = id_df.iloc[inds]
flag_df = flag_df.iloc[inds]
labels = labels[inds]

id_labels = np.column_stack((id_df.to_numpy(), labels))

for feature in features_df.columns:
	features_df[feature] = features_df[feature] - features_df[feature].mean()
	features_df[feature] = features_df[feature] / features_df[feature].std()

retain_fraction = 0.2
#need to propagate id_df through this if i want to retain names

sample, heldout, id_labels, heldout_idlabels = train_test_split(features_df, id_labels, 
			train_size=retain_fraction, stratify=id_labels[:,-1])

tsne = TSNE(n_jobs=-1)

X_embedded = tsne.fit_transform(sample.to_numpy(dtype='float'))
X_embedded_df = pd.DataFrame(X_embedded, columns=['tsne_x', 'tsne_y'])

ident = pd.DataFrame(id_labels[:,:-1], columns=['shotid','expnum','multiframe'])
labels = pd.Series(id_labels[:,-1],name='labels') 
savedf = pd.concat([ident, X_embedded_df, labels], axis=1)
savearr = np.row_stack((savedf.columns, savedf.to_numpy()))
np.save('tsne_result.npy', savearr)

