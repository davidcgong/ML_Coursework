Assignment 3 - Unsupervised Learning and Dimensionality Reduction

Datasets and analysis can be found here: https://github.com/davidcgong/ML_Coursework/tree/master/Unsupervised_Learning

All algorithms and data visualizations introduced in the analysis revolve around the usage of  WEKA, with only ICA requiring an installation of the Student Filter Package from the WEKA package manager to exclude the usage of other ICA implementations such as FastICA by sklearn.

Not all .arff datasets are included. Randomized projection and ICA was run on the original data (spambase.arff, poker-hand.arff) through the use of filters, and for the other DR algorithms, the general formatting should follow as (dataset-transformed-num_attributes-DRalgorithm), and the neural-net directory contains the data for part 5, where the data resulted from applying additional clustering on dimensionally-reduced data.
