import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from preprocessing import generate_data
from train import GRUTree, visualize


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


if __name__ == "__main__":
    import os
    import pickle

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strength', type=float, default=1000.0,
                        help='how much to weigh tree-regularization term.')
    args = parser.parse_args()

    window_length = 10


    # format is (num sequences, timesteps, features)
    # We need (features, timesteps, num sequences)

    X,Y = generate_data(9,'small_df_labeled.csv')
    obs_set, out_set = X.T,Y.T
    proportion = obs_set.shape[2]/2
    obs_train, out_train = obs_set[:, :, :proportion], out_set[:, :, :proportion]
    obs_test, out_test = obs_set[:, :, proportion:], out_set[:, :, proportion:]

    obs_train, fcpt_train, out_train = map_3d_to_2d(obs_train, out_train)
    obs_test, fcpt_test, out_test = map_3d_to_2d(obs_test, out_test)

    X_train = obs_train
    F_train = fcpt_train
    y_train = out_train

    print(X_train.shape)

    gru = GRUTree(2283, 10, [100,100,25], 1, strength=args.strength)


    gru.train(X_train, F_train, y_train, iters_retrain=50, num_iters=300,
               batch_size=10, lr=1e-2, param_scale=0.1, log_every=10)

    if not os.path.isdir('./trained_models'):
        os.mkdir('./trained_models')

    indicator = args.strength

    with open('./trained_models/trained_weights_'+str(indicator)+'.pkl', 'wb') as fp:
        pickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights}, fp)
        print('saved trained model to ./trained_models')

    visualize(gru.tree, './trained_models/tree_'+str(indicator)+'.pdf')
    print('saved final decision tree to ./trained_models')
    print('\n')

    X_test = obs_test
    F_test = fcpt_test
    y_test = out_test

    y_hat = gru.pred_fun(gru.weights, X_test, F_test)
    auc_test = roc_auc_score(y_test.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))
    avg_precision = average_precision_score(y_test.T, y_hat.T)
    print('Test Average Precision: {:.2f}'.format(avg_precision))