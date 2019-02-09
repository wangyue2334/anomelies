import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score,f1_score
import pickle
import matplotlib.pyplot as plt
from train import GRU
from adapted_fs import map_3d_to_2d

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


import os

cwd = os.getcwd()

x = np.load(cwd + '/spark_data/' + "x_test_fs.npy")
y = np.load(cwd + '/spark_data/' + "y_test_fs.npy")

print("pickle done")

with open('./trained_models/trained_weights_1000.0_fs.pkl', 'rb') as fp:
    weights = pickle.load(fp)['gru']

gru = GRU(241, 10, 1)
gru.weights = weights

X_test, F_test, y_test = map_3d_to_2d(x, y)


y_hat = gru.pred_fun(gru.weights, X_test, F_test)

auc_test = roc_auc_score(y_test.T, y_hat.T)
print('Test AUC: {:.2f}'.format(auc_test))

fpr, tpr, thresholds = roc_curve(y_test.T, y_hat.T)


# Find optimal probability threshold
threshold = Find_Optimal_Cutoff(y_test.T, y_hat.T)
print('Test threshold: {:.2f}'.format(threshold[0]))

#data['pred'] = data['pred_proba'].map(lambda x: 1 if x > threshold else 0)

import numpy as np
squarer = lambda t: 1 if t > threshold[0] else 0
vfunc = np.vectorize(squarer)
y_pred = vfunc(y_hat)

test = f1_score(y_test.T, y_pred.T)
print('Test f1-score: {:.2f}'.format(test))

unique, counts = np.unique(y_test, return_counts=True)
print(np.asarray((unique, counts)).T)

unique, counts = np.unique(y_pred, return_counts=True)
print(np.asarray((unique, counts)).T)


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.hist(y_hat[0,:], 100, normed=1, facecolor='green', alpha=0.75)
plt.show()


# Find prediction to the dataframe applying threshold
