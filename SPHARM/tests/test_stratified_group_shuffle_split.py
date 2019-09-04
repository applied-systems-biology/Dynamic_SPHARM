import unittest

import numpy as np
from ddt import ddt
from sklearn import svm
from sklearn.model_selection import cross_val_score
from scipy import ndimage

from SPHARM.classes.stratified_group_shuffle_split import GroupShuffleSplitStratified


@ddt
class TestCrossval(unittest.TestCase):

    def test_crossval(self):

        groups = np.array([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8, 9,9,9,9])
        classes = np.array([0]*9 + [1]*9 + [2]*10)
        features = np.random.rand(28,3)
        clf = svm.SVC(kernel='linear', C=1, cache_size=1000, decision_function_shape='ovo', random_state=0)
        cv = GroupShuffleSplitStratified(n_splits=5, test_size=3)
        # for train, test in cv.split(X=features, y=classes, groups=groups):
        #     print(groups[train], groups[test], classes[train], classes[test])
        #     train_classes = classes[train]
        #     unique_train_classes = np.unique(train_classes)
        #     n_observations = ndimage.sum(np.ones_like(train_classes), train_classes, unique_train_classes)
        #     predicted_classes = np.ones_like(classes[test])*unique_train_classes[np.argmax(n_observations)]
        #     print(predicted_classes)
        #     accuracy = np.sum(np.where(classes[test] == predicted_classes, 1, 0)) / len(predicted_classes)
        #     print(accuracy)
        score = cross_val_score(clf, X=features, y=classes, groups=groups, cv=cv)
        self.assertEqual(len(score), 5)


if __name__ == '__main__':
    unittest.main()



