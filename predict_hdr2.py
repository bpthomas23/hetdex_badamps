import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

print('~~~loading data~~~')
##Import data
ampstats = np.load('amp_stats_hdr2.1_notauto.npy', allow_pickle=True)
ampstats_df = pd.DataFrame(ampstats[1:], columns=ampstats[0])

print('~~~preprocessing~~~')
mask = np.any(ampstats_df == '""', axis=1)
ampstats_df = ampstats_df[~mask].reset_index()

print(np.sum(mask), ' rows dropped due to missing values.')

norm = np.load('hdr3_normalisation.npy', allow_pickle=True)
norm_df = pd.DataFrame(norm[1:,1:], index=norm[1:,0], columns=norm[0,1:])

idcols = ['shotid', 'expnum', 'multiframe']
featurecols = ['norm','N_cont','background','sky_sub_rms','wave0','wave1','nfib_bad',
		'median_sky_rms','stdev_sky_rms','sky_sub_rms_rel','im_median',
		'MaskFraction']
flagcols = ['flag_manual', 'flag','ft_flag', 'flag_badamp', 'flag_auto', 'flag_badamp_not_auto']

id_df = ampstats_df[idcols]
features_df = ampstats_df[featurecols].astype('float')
flag_df = ampstats_df[flagcols].astype('int')

"""
sel1 = (features_df['background'] > -10) & (features_df['background'] < 100)
sel2 = features_df['sky_sub_rms_rel'] < 1.5
sel3 = features_df['sky_sub_rms'] > 0.2
sel4 = (features_df['im_median'] > 0.05 ) | (np.isnan(features_df['im_median']))
sel5 = (features_df['MaskFraction'] < 0.5) | (np.isnan(features_df['MaskFraction']))
sel6 = features_df['N_cont'] < 35
sel7 = features_df['nfib_bad'] <= 1
sel8 = features_df['norm'] > 0.5

cut_flag = (sel1 & sel2 & sel3 & sel4 & sel5 & sel6 & sel7 & sel8).astype('int')
cut_flag.name = 'cut_flag'

flag_df = pd.concat([ampstats_df[flagcols], cut_flag], axis=1)
"""
#import models

print('~~~importing models~~~')
xgboost_cut_labels = joblib.load('saved_classifiers/xgboost_tuned_cut_labels_hdr3norm')
xgboost_manual_labels = joblib.load('saved_classifiers/xgboost_tuned_manual_labels_hdr3norm')
xgboost_net_labels = joblib.load('saved_classifiers/xgboost_tuned_net_labels_hdr3norm')
xgboost_notauto_labels = joblib.load('saved_classifiers/xgboost_notauto')
xgboost_notauto_good75 = joblib.load('saved_classifiers/xgboost_notauto_good75')
xgboost_notauto_hyperopt = joblib.load('saved_classifiers/xgboost_notauto_hyperopt')

norm_features_df = features_df.copy()

##normalize features to have zero mean and unit variance
for feature in norm_features_df.columns:
	norm_features_df[feature] = norm_features_df[feature] - norm_df['mean'][feature]
	norm_features_df[feature] = norm_features_df[feature] / norm_df['std'][feature]

print('~~~making predictions~~~')
pred_cut_labels = xgboost_cut_labels.predict_proba(norm_features_df)
pred_manual_labels = xgboost_manual_labels.predict_proba(norm_features_df)
pred_net_labels = xgboost_net_labels.predict_proba(norm_features_df)
pred_notauto_labels = xgboost_notauto_labels.predict_proba(norm_features_df)
pred_good75_labels = xgboost_notauto_good75.predict_proba(norm_features_df)
pred_hyperopt_labels = xgboost_notauto_hyperopt.predict_proba(norm_features_df)
#savdf = pd.concat(

###debug
#truey = flag_df['flag'].to_numpy().astype('int')
def print_confusion_matrix(truey,predy,flag):
	conf = confusion_matrix(truey,predy)
	#norm = np.sum(conf,axis=1)
	print('~~~'+flag+'~~~')
	print(conf)
#print_confusion_matrix(flag_df['flag'].to_numpy().astype('int'), 
#			np.argmax(pred_net_labels,axis=1), 'Net')
#print_confusion_matrix(flag_df['flag_manual'].to_numpy().astype('int'), 
#			np.argmax(pred_manual_labels,axis=1), 'Manual')
print_confusion_matrix(flag_df['flag'].to_numpy().astype('int'),
			np.argmax(pred_hyperopt_labels,axis=1), 'Net v. hyperopt')

#print(confusion_matrix(truey,pred_net_labels))

cut_prob = pd.Series(pred_cut_labels[:,1], name='cut_prob')
man_prob = pd.Series(pred_manual_labels[:,1], name='manual_prob')
net_prob = pd.Series(pred_net_labels[:,1], name='net_prob')
notauto_prob = pd.Series(pred_notauto_labels[:,1], name='notauto_prob')
hyperopt_prob = pd.Series(pred_hyperopt_labels[:,1], name='notauto_hyperopt_prob')

print('~~~saving predictions~~~')
savarr = pd.concat([id_df, features_df, flag_df, man_prob, net_prob, cut_prob, notauto_prob, hyperopt_prob], axis=1)
#test = pd.concat([savarr.flag, savarr.net_flag_prob],axis=1)
savstr = np.row_stack((savarr.columns,savarr.to_numpy()))
np.save('saved_predictions/predictions_hdr2.1.npy', savstr)
