"""Run this file to apply trained GRU-Tree on test set."""

from __future__ import print_function
from __future__ import absolute_import

import pickle

import autograd.numpy as np
import autograd.numpy.random as npr

from train import GRU
from sklearn.metrics import roc_auc_score,classification_report,roc_curve
from sklearn.tree import DecisionTreeClassifier
from train import average_path_length


if __name__ == "__main__":
    with open('./trained_models/trained_weights.pkl', 'rb') as fp:
        weights = pickle.load(fp)['gru']

    gru = GRU(14, 10, 1)
    gru.weights = weights

    with open('./data/test.pkl', 'rb') as fp:
        data_test = pickle.load(fp)
        X_test = data_test['X']
        F_test = data_test['F']
        y_test = data_test['y']

    y_hat = gru.pred_fun(gru.weights, X_test, F_test)
    auc_test = roc_auc_score(y_test.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))

    fpr, tpr, thresholds = roc_curve(y_test.T, y_hat.T)

    y_hat_int = np.rint(y_hat).astype(int)
    tree = DecisionTreeClassifier()



    tree.fit(X_test.T, y_hat_int.T)
    apl = average_path_length(tree, X_test.T)



    apl_count = 0
    apl_test = 0
    point_count = 0
    for i in range(X_test.shape[1]):
        test = tree.predict(X_test.T[i,:].reshape([1,14]))
        print("test= "+ str(test))
        dense_matrix = tree.decision_path(X_test.T[i, :].reshape([1, 14])).todense()
        print(dense_matrix)
        if(test==0):
            point_count +=1
            dense_matrix = tree.decision_path(X_test.T[i,:].reshape([1,14])).todense()
            print(dense_matrix)
            count = 0
            for k in range(dense_matrix.shape[1]):
                if(dense_matrix[0, k]==1):
                    count +=1
            print(count)
            apl_count = apl_count + count

    average_apl = float(apl_count)/point_count
    print(float(apl_test)/point_count)
    print("average apl for anomalies: " + str(average_apl))
    print("average apl: " + str(apl))
