# hetdex_badamps
Supervised learning to identify bad data in HETDEX

HETDEX is a revolutionary astronomical survey that is collecting GBs of data every night to measure the parameters pertaining to dark energy, and thereby probe the accelerating expansion of the Universe. To do this, HETDEX aims to collect spectra of millions of galaxies, from nearby star-forming galaxies to distant galaxies with bright, active nuclei whose luminosities are produced by matter orbiting around their central supermassive black holes.

The enormous data collection abilities of HETDEX come with their challenges. Bad data can be difficult to find in a catalogue of two million individual samples. In past survey iterations, these bad data have had to be found by hand, with experts spending weeks or months trawling through the massive data-set to identify images where the hardware has failed (amongst a host of other potential issues). These historical data-sets provide the perfect training set for supervised machine learning.

Using twelve features across nearly two million samples, I have investiagted the relative efficacies of a host of supervised machine learning algorithms, including XGBoost, random forest, support vector machines, K-nearest neighbours, and neural networks. I found that before hyperparameter tuning, XGBoost was the best performing of these algorithms in correctly predicting whether a held-out test-set of samples were labelled as good or bad in the historical data set of 1.8 million objects. XGBoost was thus selected for further tuning.

XGBoost is also relatively fast to train, taking about 20 minutes on the full data-set. I tuned the algorithm by hand, selecting a small but representative sample of a few tens of thousands of samples and iteratively training on different selections of hyperparameters to find the best performing result as quantified by the area under the receiver operating characterics (ROC) curve. The benefit of training by hand is that the human guide learns in which directions in the hyperparameter space the performace metric is likely to improve; humans can select promising hyperparameter combinations on-the-fly. This contrasts to popular hyperparaemter optimization routines such as a full grid search or a randomized grid search which are blind to the shape of the loss function in between training iterations. An alternative hyperparameter optimization routine is bayseian optimization, which `learns` the shape of the loss function as it explores the hyperparameter space, and uses a surrogate model of the loss function which is iteratively updated during hyperparameter exploration to also find promising hyperparameter combinations.

%%results

%%limitations

%%future work
