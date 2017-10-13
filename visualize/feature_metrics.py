import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from visualize.feature_plots import load_unsw_feature_names


def autolabel(rects, label):
    """Attach a text label above each bar displaying its height"""
    for (i, rect) in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                 '%s' % label[i],
                 ha='center', va='bottom')


def plot_feature_importance(X, y, dirname, prefix):
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  criterion='entropy',
                                  random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names = load_unsw_feature_names()
    sorted_features = [feature_names[i] for i in indices]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("Feature %s(%d) = %f" % (feature_names[indices[f]],
                                       indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature Importances")
    rects = plt.bar(range(X.shape[1]), importances[indices],
                    color="b", yerr=std[indices], ecolor='r', align="center")
    autolabel(rects, sorted_features)
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig('%s/%s_feature_importance.png' % (dirname, prefix), dpi=400)
    plt.show()


def plot_pca_components(X, y, dirname, prefix):
    pca = PCA(n_components=None, random_state=0)
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    importance = pca.explained_variance_ratio_
    indices = np.argsort(eigenvalues)[::-1]

    print('PCA ranking:')
    for f in range(X.shape[1]):
        print("%d. eigenvalue = %f" % (f, eigenvalues[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("PCA Components Ratio")
    plt.bar(range(X.shape[1]), importance[indices],
            color="b", align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig('%s/%s_principle_components.png' % (dirname, prefix), dpi=400)
    plt.show()
