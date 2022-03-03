# hetdex_badamps
Supervised learning to identify bad data in HETDEX

 HETDEX is a revolutionary astronomical survey that is collecting GBs of data every night to measure the parameters pertaining to dark energy, 
 and thereby probe the accelerating expansion of the Universe. To do this, HETDEX aims to collect spectra of millions of galaxies, from nearby 
 star-forming galaxies to distant galaxies with bright, active nuclei whose luminosities are produced by matter orbiting around their central 
 supermassive black holes.

 The enormous data collection abilities of HETDEX come with their challenges. Bad data can be difficult to find in a catalogue of two million 
 individual samples. In past survey iterations, these bad data have had to be found by hand, with experts spending weeks or months trawling 
 through the massive data-set to identify images where the hardware has failed (amongst a host of other potential issues). These historical 
 data-sets provide the perfect training set for supervised machine learning.

 Using twelve features across nearly two million samples, I have investiagted the relative efficacies of a host of supervised machine learning 
 algorithms, including XGBoost, random forest, support vector machines, K-nearest neighbours, and neural networks. I found that before hyperparameter 
 tuning, XGBoost was the best performing of these algorithms in correctly predicting whether a held-out test-set of samples were labelled as good or 
 bad in the historical data set of 1.8 million objects. XGBoost was thus selected for further tuning.

 XGBoost is also relatively fast to train, taking about 20 minutes on the full data-set (depending on the specific set of hyperparameters). After a 
 first iteration of tuning the algorithm by hand, I decided to use a Bayesian algorithm to find the best optimization of the hyperparameters.
 Bayseian optimization of hyperparameters is implemented in the Python package hyperopt. This method of tuning machine learning algorithms is 
 superior to traditionl methods such a a full grid search or randomized grid search as it decides on the next set of hyperparameters to try
 based on the resulting loss of all previous attempts to find the optimal set; it effecively 'learns' the shape of the loss function as it tunes.
 
 Using the above method I achieved an AUC score of >99%, indiciating that this data-set is relatively easy to predict. A factor that contributed 
 to this extremely high score was a careful selection of the training set of 100K samples; we were specifically aiming to find bad amplifiers that
 were not found using other automated methods. I was able to save the HETDEX team of astronomy experts months of having to manually trawl through the
 data and classify bad amplifiers by hand.
 
 The code contained in this repo represents a handful of key scripts that demonstrate the workflow of the project. Data-files and model-files are 
 omitted.

