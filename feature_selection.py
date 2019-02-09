from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import os

cwd = os.getcwd()

x = np.load(cwd + '/spark_data/' + "x.npy")
y = np.load(cwd + '/spark_data/' + "y.npy")

print("pickle done")

X, Y = x[:,0,:], y[:,0,:].reshape((y.shape[0]))
print(X.shape)
clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)
print(clf.feature_importances_)

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
