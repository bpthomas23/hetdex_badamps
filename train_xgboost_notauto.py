import numpy as np
#from build_feature_classifier import BinaryClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm #this is a hunk of junk
import xgboost as xgb
import joblib
import pandas as pd

#calfib_chunk = np.load('/work/06567/bpthomas/stampede2/badamps_project/calfib_chunks/calfib_0_400.npy')
#calfib_chunk = np.transpose(calfib_chunk, (2, 0, 1))
#info_chunk = np.load('/work/06567/bpthomas/stampede2/badamps_project/calfib_chunks/info_0_400.npy')
print('~~~loading data~~~')
#amp_stats = np.loadtxt('amp_stats.txt', str)

amp_stats = np.load('amp_stats_hdr2.1_notauto.npy',allow_pickle=True)
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
#kinda janky
features = features.drop(['Date', 'date'], axis=1)

#now I can convert the whole features array into float
features = features.astype('float')

#choose from flag_manual, flag, ft_flag, cut_flag as the target variable
#choosing automated flag to start with as it is probably easiest to reproduce
labels = ~flags['flag_badamp_not_auto'].astype('bool')

labels = labels.astype('int')

#normalise every feature to have zero mean and unit variance
for feature_name in features.columns:
	features[feature_name] = features[feature_name] - norms_df['mean'][feature_name]
	features[feature_name] = features[feature_name] / norms_df['std'][feature_name]


#keep only as much good data as there is bad data, such that the classes have a 50/50 split
np.random.seed(42)
goodinds = np.where(labels == 1)[0]
badinds = np.where(labels == 0)[0]
newgoodinds = np.random.choice(goodinds, size=len(badinds), replace=False)
inds = np.append(badinds,newgoodinds)
np.random.shuffle(inds)

features = features.iloc[inds]
labels = labels.iloc[inds]
identifiers = identifiers.iloc[inds]

print(np.sum(labels)/len(labels) * 100, ' percent of the data is good')

"""i can ignore this evaluation option as the model never sees the test set testX,testy during training, it splits the train set trainX,trainy
into train and test sets. but those are all made from the train set i defined: testX,testy are held out throughout"""


#take 30% of the data for an independent evaluation later
#inputX,evalX,inputy,evaly=train_test_split(features,labels,test_size=0.3,random_state=12,stratify=labels)


inputX = features
inputy = pd.concat([identifiers,labels], axis=1)
trainX,testX,trainy,testy=train_test_split(inputX,inputy,test_size=0.3,random_state=12,stratify=inputy.iloc[:,3].astype('int'))

trainid = trainy.iloc[:,:3]
trainy = trainy.iloc[:,3].astype('int')
testid = testy.iloc[:,:3]
testy = testy.iloc[:,3].astype('int')

#initialize param dict

#learning rate = 1
#n_estimators = 1600 -- improvement flattens to < 0.01% above that
#gamma = 0
#max_depth = 10 -- improvement flattens to < 0.01% above that
#min_child_weight = 0
#colsample_bytree = 1
params = {'learning_rate' : [1] , 
	  'n_estimators': [1600],
	  'gamma': [0],
          'max_depth': [10],
          'min_child_weight': [0],
	  'colsample_bytree': [1],
	  'lambda': [1],
	  'alpha': [0]}


#classifier = RandomForestClassifier(verbose=True,n_jobs=1)
classifier = xgb.XGBClassifier(n_jobs=-1, objective='binary:logistic', verbosity=1, eval_metric='auc', tree_method='hist', n_estimators=1600, max_depth=10)

#keep this just for cross-validation
grid_auc = GridSearchCV(param_grid=params, estimator=classifier, scoring='roc_auc', cv=4, verbose=1, n_jobs=4)

grid_auc.fit(trainX,trainy)
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
np.save('saved_predictions/'+model+'.npy', np.column_stack((testid,testy,predy_proba)))
