import unittest

import numpy as np
from ddt import ddt
from sklearn import svm
from sklearn.model_selection import cross_val_score

from SPHARM.classes.stratified_group_shuffle_split import GroupShuffleSplitStratified


@ddt
class TestCrossval(unittest.TestCase):

    def test_crossval(self):

        groups = np.array([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8, 9,9,9])
        classes = np.array([0]*9 + [1]*9 + [2]*9)
        features = np.random.rand(27,3)
        clf = svm.SVC(kernel='linear', C=1, cache_size=1000, decision_function_shape='ovo', random_state=0)
        cv = GroupShuffleSplitStratified(n_splits=5, test_size=3)
        for train, test in cv.split(X=features, y=classes, groups=groups):
            print(groups[train], groups[test], classes[train], classes[test])
        score = cross_val_score(clf, X=features, y=classes, groups=groups, cv=cv)
        self.assertEqual(len(score), 5)


if __name__ == '__main__':
    unittest.main()



