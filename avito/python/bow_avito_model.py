import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV
from sklearn.utils import parallel_backend
from sklearn.model_selection import train_test_split


# T is the length of data you want to run right now (the full dataset takes forever
# mdf is for setting the min_df value in the TfidfVectorizer function (google is good) -- when building the vocabulary for back of words ignore terms that have a document frequency strictly lower than min_df
T = 30000
mdf = 50

# get data from csv files
data = pd.read_csv('train.csv', usecols=['description', 'deal_probability'])
desc = (data['description'])
Y = (data['deal_probability'])
data = 0

# break up data into train and test data
traindesc, testdesc, trainY, testY = train_test_split(desc, Y, test_size=0.25, random_state=23)

# shrink training data to T
traindesc = traindesc[:T]
trainY = trainY[:T]

# Replace nans with spaces
traindesc.fillna(" ", inplace=True)
testdesc.fillna(" ", inplace=True)

## Get "bag of words" transformation of the data -- see example in Lasso book discussed in class
## also: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
vec = TfidfVectorizer(ngram_range=(1,1),
                      min_df=mdf,
                      max_df=0.9,
                      lowercase=True,
                      strip_accents='unicode',
                      sublinear_tf=True)
trainX = vec.fit_transform(traindesc)
testX = vec.transform(testdesc)

# fit lasso model
with(parallel_backend('threading')):
    m = LassoCV(cv=5, verbose=True).fit(trainX, trainY)

# show results for fit data
plt.figure()
ax = plt.subplot(111)
plt.plot(m.alphas_, m.mse_path_, ':')
plt.plot(m.alphas_, m.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(m.alpha_, linestyle='--', color='k', label='CV estimate')
ax.set_xscale('log')
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('MSE')
plt.axis('tight')
plt.savefig('lasso_path.png')

# show the terrible predictions
testYpred = m.predict(testX)
plt.figure()
plt.plot(testY, testYpred, '.', alpha=0.1)
plt.title('RMSE: %f' % np.sqrt(np.mean((testYpred - testY) ** 2)))
plt.savefig('lasso_prediction.png')
