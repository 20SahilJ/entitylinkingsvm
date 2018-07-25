import itertools
import numpy as np
from scipy import stats
import matplotlib.pylab as pl
from scipy._lib._ccallback_c import idx
from sklearn import svm, linear_model, cross_validation

np.random.seed(0)
theta = np.deg2rad(60)
w = np.array([np.sin(theta), np.cos(theta)])
K = 20
X = np.random.randn(K, 2)
y = [0] * K
for i in range(1, 3):
    X = np.concatenate((X, np.random.randn(K, 2) + i * 4 * w))
    y = np.concatenate((y, [i] * K))


# slightly displace data corresponding to our second partition
X[::2] -= np.array([3, 7])
blocks = np.array([0, 1] * int((X.shape[0]) / 2))

# split into train and test set
cv = cross_validation.StratifiedShuffleSplit(y, test_size=.5)
train, test = iter(cv).__next__()
X_train, y_train, b_train = X[train], y[train], blocks[train]
X_test, y_test, b_test = X[test], y[test], blocks[test]


#print("Got here")

comb = itertools.combinations(range(X_train.shape[0]), 2)
k = 0
Xp, yp, diff = [], [], []
for (i, j) in comb:
    if y_train[i] == y_train[j] \
        or blocks[train][i] != blocks[train][j]:
        # skip if same target or different group
        continue
    Xp.append(X_train[i] - X_train[j])
    diff.append(y_train[i] - y_train[j])
    yp.append(np.sign(diff[-1]))
    # output balanced classes
    if yp[-1] != (-1) ** k:
        yp[-1] *= -1
        Xp[-1] *= -1
        diff[-1] *= -1
    k += 1
Xp, yp, diff = map(np.asanyarray, (Xp, yp, diff))
clf = svm.SVC(kernel='linear', C=.1)
#print(yp[0])
#print(yp[1])
#print(yp)
clf.fit(Xp, yp)
coef = clf.coef_.ravel() / pl.linalg.norm(clf.coef_)
#print(coef)
#print(coef)
#print(X_test)
#print(y_test)
#print("Got here 2")
for i in range(2):
    tau, _ = stats.kendalltau(
        np.dot(X_test[b_test == i], coef), y_test[b_test == i])
    #print('Kendall correlation coefficient for block %s: %.5f' % (i, tau))


