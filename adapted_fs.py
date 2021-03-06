import autograd.numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,roc_curve,f1_score,precision_score
from train import GRUTree, visualize,average_path_length
import pandas as pd
import time



def map_3d_to_2d(X_arr, y_arr):
    """Convert 3d NumPy array to 2d NumPy array with fenceposts.

    @param X_arr: NumPy array
                  size D x T x N
    @param t_arr: NumPy array
                  size O x T x N
    @return X_arr_DM: NumPy array
                      size D x (T x N)
    @return y_arr_DM: NumPy array
                      size O x (T x N)
    @return fenceposts_Np1: NumPy array (1-dimensional)
                            represents splits between sequences.
    """
    n_in_dims, n_timesteps, n_seqs = X_arr.shape
    n_out_dims, _, _ = y_arr.shape

    X_arr_DM = X_arr.swapaxes(0, 2).reshape((-1, n_in_dims)).T
    y_arr_DM = y_arr.swapaxes(0, 2).reshape((-1, n_out_dims)).T

    fenceposts_Np1 = np.arange(0, (n_seqs + 1) * n_timesteps, n_timesteps)
    return X_arr_DM, fenceposts_Np1, y_arr_DM

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

if __name__ == "__main__":
    import os
    import pickle

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strength', type=float, default=1000.0,
                        help='how much to weigh tree-regularization term.')
    args = parser.parse_args()

    window_length = 10

    import os

    cwd = os.getcwd()

    x_train = np.load(cwd + '/spark_data/' + "x_train_mixed_fs.npy")
    y_train = np.load(cwd + '/spark_data/' + "y_train_mixed_fs.npy")
    x_test = np.load(cwd + '/spark_data/' + "x_test_mixed_fs.npy")
    y_test = np.load(cwd + '/spark_data/' + "y_test_mixed_fs.npy")

    print(x_train.shape)

    obs_train, fcpt_train, out_train = map_3d_to_2d(x_train, y_train)
    obs_test, fcpt_test, out_test = map_3d_to_2d(x_test, y_test)

    print(obs_train.shape)

    X_train = obs_train
    F_train = fcpt_train
    y_train = out_train

    number_of_features = 215

    gru = GRUTree(number_of_features, 10, [25], 1, strength=args.strength)


    gru.train(X_train, F_train, y_train, iters_retrain=20, num_iters=1500,
               batch_size=10, lr=1e-3, param_scale=0.1, log_every=1)

    tree = gru.tree

    if not os.path.isdir('./trained_models'):
        os.mkdir('./trained_models')

    indicator = str(args.strength)

    with open('./trained_models/trained_weights_'+indicator+'_mixed_fs2.pkl', 'wb') as fp:
        pickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights}, fp)
        print('saved trained model to ./trained_models')

    visualize(gru.tree, './trained_models/tree_'+str(indicator)+'_mixed_fs2.pdf',True)
    print('saved final decision tree to ./trained_models')
    print('\n')

    print('name of the file: ./trained_models/trained_weights_'+indicator+'_mixed_fs2.pkl')

    X_test = obs_test
    F_test = fcpt_test
    y_test = out_test

    y_hat = gru.pred_fun(gru.weights, X_test, F_test)
    y_hat_int = np.rint(y_hat).astype(int)
    auc_test = roc_auc_score(y_test.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))
    avg_precision = average_precision_score(y_test.T, y_hat.T)
    print('Test Average Precision: {:.2f}'.format(avg_precision))

    fpr, tpr, thresholds = roc_curve(y_test.T, y_hat.T)

    # Find optimal probability threshold
    threshold = Find_Optimal_Cutoff(y_test.T, y_hat.T)
    print('Test threshold: {:.2f}'.format(threshold[0]))

    # data['pred'] = data['pred_proba'].map(lambda x: 1 if x > threshold else 0)

    import numpy as np

    squarer = lambda t: 1 if t > threshold[0] else 0
    vfunc = np.vectorize(squarer)
    y_pred = vfunc(y_hat)

    test = f1_score(y_test.T, y_pred.T)
    print('Test f1-score: {:.2f}'.format(test))

    print('\n')
    print('\n')

    y_tree = tree.predict(X_test.T)
    apl = average_path_length(tree, X_test.T)
    print('APL for tree: {:.2f}'.format(apl))
    #auc_tree = roc_auc_score(y_test.T, y_tree.T)
    #print('Test AUC for tree: {:.2f}'.format(auc_tree))
    y_pred_tree = vfunc(y_tree)
    test_tree = f1_score(y_hat_int.T, y_pred_tree.T)
    print('Test f1-score for tree: {:.2f}'.format(test_tree))
    precision_tree = precision_score(y_hat_int.T,y_pred_tree.T)
    print('Test precision for tree: {:.2f}'.format(precision_tree))

    apl_count = 0
    point_count = 0
    max_path = 0
    min_path = 1000000
    leaf_list = []
    for i in range(X_test.shape[1]):
        if (i % 10000 == 0):
            print('iteration count= ' + str(i))
        test = tree.predict(X_test.T[i, :].reshape([1, number_of_features]))
        # Label = 1 means anomaly
        if (test == 1):
            leaf_final = tree.apply(X_test.T[i, :].reshape([1, number_of_features]))
            if leaf_final not in leaf_list:
                leaf_list.append(leaf_final)
            point_count += 1
            dense_matrix = tree.decision_path(X_test.T[i, :].reshape([1, number_of_features])).todense()
            count = 0
            for k in range(dense_matrix.shape[1]):
                if (dense_matrix[0, k] == 1):
                    count += 1
            if (count > max_path):
                max_path = count
            if (count < min_path):
                min_path = count
            apl_count = apl_count + count

    average_apl = float(apl_count) / point_count
    print("number of paths: " + str(len(leaf_list)))
    print("APL for anomalies: " + str(average_apl))
    print("Max path length for anomaly: " + str(max_path))
    print("Min path length for anomaly: " + str(min_path))