print('~~~loading libraries~~~')
import numpy as np
#from build_feature_classifier import BinaryClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm #this is a hunk of junk
import xgboost as xgb
import joblib
import pandas as pd
from hyperopt import fmin, tpe, hp

print('~~~loading data~~~')

amp_stats = np.load('amp_stats_hdr2.1_20220303.npy',allow_pickle=True)
amp_stats_df = pd.DataFrame(amp_stats[1:], columns=amp_stats[0])

##load normalizations:
norms = np.load('hdr3_normalisation.npy',allow_pickle=True)
norms_df = pd.DataFrame(norms[1:,1:], index=norms[1:,0], columns=norms[0,1:])


print('~~~preprocessing~~~')

identifiers = amp_stats_df.iloc[:,:3]
flags = amp_stats_df.iloc[:,-6:]
features = amp_stats_df.iloc[:,3:-6]

mask = features.MaskFraction == '""'

features = features[~mask]
flags = flags[~mask]
identifiers = identifiers[~mask]

#i also want to drop two ``date`` features from features; cols 7 and 11

features = features.drop(['Date', 'date'], axis=1)

#now I can convert the whole features array into float
features = features.astype('float')

#choose from flag_manual, flag, ft_flag, cut_flag as the target variable
labels = ~flags['flag_notauto'].astype('bool')
labels = labels.astype('int')
net_labels = flags['flag'].astype('int')

#normalise every feature to have zero mean and unit variance
for feature_name in features.columns:
	features[feature_name] = features[feature_name] - norms_df['mean'][feature_name]
	features[feature_name] = features[feature_name] / norms_df['std'][feature_name]


#keep only as much good data as there is bad data, such that the classes have a 50/50 split
np.random.seed(42)
goodinds = np.where(net_labels == 1)[0]
badinds = np.where(labels == 0)[0]
newgoodinds = np.random.choice(goodinds, size=len(badinds), replace=False)
inds = np.append(badinds,newgoodinds)
np.random.shuffle(inds)

features = features.iloc[inds]
labels = labels.iloc[inds]
identifiers = identifiers.iloc[inds]

print(np.sum(labels)/len(labels) * 100, ' percent of the data is good')


inputX = features
inputy = pd.concat([identifiers,labels], axis=1)
trainX,testX,trainy,testy=train_test_split(inputX,inputy,test_size=0.3,random_state=12,stratify=inputy.iloc[:,3].astype('int'))

trainid = trainy.iloc[:,:3]
trainy = trainy.iloc[:,3].astype('int')
testid = testy.iloc[:,:3]
testy = testy.iloc[:,3].astype('int')

##taking subset of data for testing
#trainX = trainX[:10000]
#trainy = trainy[:10000]

#initialize param dict

#learning rate = 1
#n_estimators = 1600 -- improvement flattens to < 0.01% above that
#gamma = 0
#max_depth = 10 -- improvement flattens to < 0.01% above that
#min_child_weight = 0
#colsample_bytree = 1

##define search space

space = {'learning_rate' : hp.uniform('learning_rate', 0.01, 1), 
	  'n_estimators': hp.uniformint('n_estimators', 100, 500),
	  'gamma': hp.uniform('gamma', 0, 100),
          'max_depth': hp.uniformint('max_depth', 3, 20),
          'min_child_weight': hp.uniform('min_child_weight', 0, 100),
	  'colsample_bytree': hp.uniform('colsample_bytree', 1/12, 1),
	  'lambda': hp.uniform('lambda', 1, 50),
	  'alpha': hp.uniform('alpha', 0, 50)}

def evaluate(hp_space):
	learning_rate = hp_space['learning_rate']
	n_estimators = int(hp_space['n_estimators'])
	gamma= hp_space['gamma']
	max_depth = int(hp_space['max_depth'])
	min_child_weight = hp_space['min_child_weight']
	colsample_bytree = hp_space['colsample_bytree']
	lamb = hp_space['lambda']
	alpha = hp_space['alpha'] 

	classifier = xgb.XGBClassifier(n_jobs=-1, objective='binary:logistic', verbosity=1, eval_metric='auc', tree_method='hist', learning_rate=learning_rate, n_estimators=n_estimators, gamma=gamma, max_depth=max_depth, min_child_weight=min_child_weight, colsample_bytree=colsample_bytree, reg_lambda=lamb, alpha=alpha)

	#keep this just for cross-validation
	cv = cross_val_score(classifier, trainX, y=trainy, scoring='roc_auc', cv=4, verbose=1, n_jobs=-1)

	best = np.max(cv)
	return -best #minus sign as hyperopt minimizes

best_hp = fmin(fn=evaluate, space=space, algo=tpe.suggest, max_evals=50)
pd.Series(best_hp).to_csv('best_hyperparameters_20220303.csv', header=False)

"""
model = 'xgboost_notauto'

best = grid_auc.best_estimator_
feature_importances = best.feature_importances_
impidx = feature_importances.argsort()
plt.barh(features.columns[impidx], feature_importances[impidx])
plt.xlabel('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('metric_plots/feature_importance'+model+'.pdf')
plt.close()


#classifier = svm.SVC(probability=True)
#classifier = KNeighborsClassifier(n_jobs=4)

predy_proba = best.predict_proba(testX)
predy = np.argmax(predy_proba, axis=1)

confusion_matrix = confusion_matrix(testy, predy)
classification_report = classification_report(testy,predy)
print(confusion_matrix)
print(classification_report)

#import pickle
#with open('saved_classifiers/'+model+'.pickle','wb') as savefile:
#	pickle.dump(classifier,savefile)

joblib.dump(best, 'saved_classifiers/'+model)
#best.save_model(model+'.json')
np.save('saved_predictions/'+model+'.npy', np.column_stack((testid,testy,predy_proba)))"""
