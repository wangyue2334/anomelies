import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


# convert series to supervised learning
def series_to_supervised(data, n_in=10, n_out=1, dropnan=True):
    df = data
    n_vars = data.shape[1]
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(str(df.keys()[j]) + str('(t-%d)' % (i))) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(str(df.keys()[j]) + str('(t-%d)' % (i))) for j in range(n_vars)]
        else:
            names += [(str(df.keys()[j]) + str('(t-%d)' % (i))) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # agg = agg.drop(['anomaly(t-1)','anomaly(t-2)','anomaly(t-3)','anomaly(t-4)','anomaly(t-5)','anomaly(t-6)','anomaly(t-7)','anomaly(t-8)','anomaly(t-9)','anomaly(t-10)'],axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def generate_dataframe(window_length, filename):
    # load dataset

    import os
    cwd = os.getcwd()
    dataset = pd.read_csv(cwd + '/spark_data/' + filename, header=0, index_col='t')

    dataset = dataset.fillna(-1)
    # integer encode direction
    # encoder = LabelEncoder()
    # values[:, 4] = encoder.fit_transform(values[:, 4])
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
    # ensure all data is float
    scaled = df_scaled.astype('float64')
    # frame as supervised learning
    reframed = series_to_supervised(scaled, window_length, 1)
    label_df = reframed[
        ['anomaly(t-0)', 'anomaly(t-1)', 'anomaly(t-2)', 'anomaly(t-3)', 'anomaly(t-4)', 'anomaly(t-5)', 'anomaly(t-6)',
         'anomaly(t-7)', 'anomaly(t-8)', 'anomaly(t-9)']]
    reframed = reframed.drop(
        ['anomaly(t-0)', 'anomaly(t-1)', 'anomaly(t-2)', 'anomaly(t-3)', 'anomaly(t-4)', 'anomaly(t-5)', 'anomaly(t-6)',
         'anomaly(t-7)',
         'anomaly(t-8)', 'anomaly(t-9)'], axis=1)

    feature_name = reframed.keys()
    with open('./feature_name/list.pkl', 'wb') as fp:
        pickle.dump(feature_name, fp)
        # print('saved feature names in /feature_name')

    return reframed, label_df


def generate_dataframe_feature_selection(window_length, filename, feature_selection_model=None):
    # load dataset

    import os
    cwd = os.getcwd()
    dataset = pd.read_csv(cwd + '/spark_data/' + filename, header=0, index_col='t')
    print(filename)
    print(dataset.shape)

    dataset = dataset.fillna(-1)
    # integer encode direction
    # encoder = LabelEncoder()
    # values[:, 4] = encoder.fit_transform(values[:, 4])
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
    # ensure all data is float
    scaled = df_scaled.astype('float64')

    # To balance the classes
    abnormal = scaled[scaled['anomaly'] == 1].shape[0]
    normal = scaled[scaled['anomaly'] == 0].shape[0]

    alpha = 1 - (float(abnormal) / normal)
    print(alpha)
    if (alpha >= 0):
         scaled = scaled.drop(scaled[scaled['anomaly'] == 0].sample(frac=alpha).index)
         print(scaled[scaled['anomaly'] == 1].shape[0])
         print(scaled[scaled['anomaly'] == 0].shape[0])
    if (alpha < 0):
         alpha_prime = 1 - (float(normal) / abnormal)
         scaled = scaled.drop(scaled[scaled['anomaly'] == 1].sample(frac=alpha_prime).index)
         print(scaled[scaled['anomaly'] == 1].shape[0])
         print(scaled[scaled['anomaly'] == 0].shape[0])

    # y_f = (abnormal*0.7)/(0.3)
    # print(y_f)
    # alpha = (float(normal)-y_f)/normal
    # print(alpha)
    # if(alpha > 0):
    #     scaled = scaled.drop(scaled[scaled['anomaly'] == 0].sample(frac=alpha).index)
    # else:
    #     y_f = (normal * 0.3) / (0.7)
    #     print(y_f)
    #     alpha = (float(abnormal) - y_f) / abnormal
    #     print(alpha)
    #     scaled = scaled.drop(scaled[scaled['anomaly'] == 1].sample(frac=alpha).index)

    if (feature_selection_model is None):  # Meant for the dataset with both normal and abnormal

        clf = ExtraTreesClassifier()
        clf = clf.fit(scaled.loc[:, scaled.columns != 'anomaly'], scaled['anomaly'])
        # (clf.feature_importances_)

        model = SelectFromModel(clf, prefit=True)
        #model.transform(scaled.loc[:, scaled.columns != 'anomaly'])
        new_scaled = scaled.loc[:, scaled.columns]
        #new_scaled['anomaly'] = scaled.loc[:, 'anomaly']

        # print(scaled.shape)
        # print(new_scaled.shape)
        # print(new_scaled)
        model_out = model
    else:  # Meant for the dataset with only abnormal instances
        #feature_selection_model.transform(scaled.loc[:, scaled.columns != 'anomaly'])
        #new_scaled = scaled.loc[:, feature_selection_model.get_support(indices=False)]
        #new_scaled['anomaly'] = scaled.loc[:, 'anomaly']
        # print(scaled.shape)
        # print(new_scaled.shape)
        # print(new_scaled)
        model_out = None
        #Remove for feature engineering
        new_scaled = scaled.loc[:, scaled.columns]

    # frame as supervised learning
    reframed = series_to_supervised(new_scaled, window_length, 1)
    label_df = reframed[
        ['anomaly(t-0)', 'anomaly(t-1)', 'anomaly(t-2)', 'anomaly(t-3)', 'anomaly(t-4)', 'anomaly(t-5)', 'anomaly(t-6)',
         'anomaly(t-7)', 'anomaly(t-8)', 'anomaly(t-9)']]
    reframed = reframed.drop(
        ['anomaly(t-0)', 'anomaly(t-1)', 'anomaly(t-2)', 'anomaly(t-3)', 'anomaly(t-4)', 'anomaly(t-5)', 'anomaly(t-6)',
         'anomaly(t-7)',
         'anomaly(t-8)', 'anomaly(t-9)'], axis=1)

    feature_name = reframed.keys()
    with open('./feature_name/list_feature_selection.pkl', 'wb') as fp:
        pickle.dump(feature_name, fp)
        # print('saved feature names in /feature_name')

    return reframed, label_df, model_out


def generate_data_and_feature_selection(window_length, filename, feature_selection_model=None):
    reframed, labels, model = generate_dataframe_feature_selection(window_length, filename, feature_selection_model)
    # num_features per timestep
    num_features = (reframed.shape[1]) / (window_length + 1)
    # split into train and test sets
    values = reframed
    labels = labels.values
    window_list = []
    windowed_labels = []
    for i in range(values.shape[0]):
        timestep_list = []
        label_list = []
        for j in range(window_length + 1):
            label_list.append(labels[i, j])
            timestep_list.append(values.iloc[i, j:j + num_features])
        window = np.stack(timestep_list, axis=0)
        window_list.append(window)
        step = np.stack(label_list, axis=0)
        windowed_labels.append(step)

    x_set = np.stack(window_list, axis=0)
    y_set = np.stack(windowed_labels, axis=0)
    y_set = y_set.reshape((y_set.shape[0], y_set.shape[1], 1))

    # print("number of sequences= "+ str(x_set.shape[0]))

    return x_set, y_set, model


def generate_data(window_length, filename):
    reframed, labels = generate_dataframe(window_length, filename)
    # num_features per timestep
    num_features = (reframed.shape[1]) / (window_length + 1)
    # split into train and test sets
    values = reframed
    labels = labels.values
    window_list = []
    windowed_labels = []
    for i in range(values.shape[0]):
        timestep_list = []
        label_list = []
        for j in range(window_length + 1):
            label_list.append(labels[i, j])
            timestep_list.append(values.iloc[i, j:j + num_features])
        window = np.stack(timestep_list, axis=0)
        window_list.append(window)
        step = np.stack(label_list, axis=0)
        windowed_labels.append(step)

    x_set = np.stack(window_list, axis=0)
    y_set = np.stack(windowed_labels, axis=0)
    y_set = y_set.reshape((y_set.shape[0], y_set.shape[1], 1))

    print("number of sequences= " + str(x_set.shape[0]))

    return x_set, y_set


#filename_list = ['20_51_labeled.csv', '20_52_labeled.csv', '20_53_labeled.csv',
#                   '66_5_labeled.csv', '66_7_labeled.csv', '66_8_labeled.csv', '76_1_labeled.csv',
#                  '76_2_labeled.csv', '146_17_labeled.csv', '146_18_labeled.csv',
#                   '146_20_labeled.csv']
#
filename_list = ['146_17_labeled.csv', '146_18_labeled.csv',
                   '146_20_labeled.csv']
df = generate_data_and_feature_selection(9, '146_19_labeled.csv')
unique, counts = np.unique(df[1], return_counts=True)
print("main: ")
print(np.asarray((unique, counts)).T)
# # #
df_list = [df]
# # #
for filename in filename_list:
       df1 = generate_data_and_feature_selection(9, filename, df[2])
       unique, counts = np.unique(df1[1], return_counts=True)
       print("second: ")
       print(np.asarray((unique, counts)).T)
       df_list.append(df)
x = np.concatenate([df0[0] for df0 in df_list])
y = np.concatenate([df0[1] for df0 in df_list])
# # # #
print(x.shape)
print(y.shape)
# # #
unique, counts = np.unique(y, return_counts=True)
print("final: ")
print(np.asarray((unique, counts)).T)
# # #
import os
# # #
cwd = os.getcwd()
# # #
idxs = np.arange(x.shape[0])
# # #
y_split = y.reshape((y.shape[0], y.shape[1]))
idxs_train, idxs_test, y_train, y_test = train_test_split(idxs, y_split, test_size=0.33, random_state=42, shuffle=True,stratify=y_split)
print(idxs_train.shape)
print(idxs_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train, x_test, y_train, y_test = x[idxs_train, :, :].T, x[idxs_test, :, :].T, y_train.reshape(
       (y_train.shape[0], y_train.shape[1], 1)).T, y_test.reshape((y_test.shape[0], y_test.shape[1], 1)).T
# # # #
# # # # #
print("split done")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# # # # #
np.save(cwd + '/spark_data/' + "x_train_fullv2.npy", x_train)
np.save(cwd + '/spark_data/' + "y_train_fullv2.npy", y_train)
np.save(cwd + '/spark_data/' + "x_test_fullv2.npy", x_test)
np.save(cwd + '/spark_data/' + "y_test_fullv2.npy", y_test)
