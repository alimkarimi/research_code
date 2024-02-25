# test to import gdal:
from osgeo import gdal

def final_test():
    print('found this too!!!')

li = [0,1,2,3,5]
print(li[1:-1])

#standard scaling
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np

# 4 samples/observations and 2 variables/features
# data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# mean1 = data[:,0].mean()
# mean2 = data[:,1].mean()
# std1 = data[:,0].std()
# std2 = data[:,1].std()
# print(mean1, mean2, std1, std2)
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)
# print(data.shape)
# print(data)
# print(scaled_data)

from sklearn.model_selection import StratifiedShuffleSplit
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7,7], [8,8], [9,9], [10, 10], [11, 11], [12, 12], [13,13], [14, 14], [15, 15], [16,16]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
sss = StratifiedShuffleSplit(n_splits=5, test_size=6/12, random_state=1)
print(sss.get_n_splits(X, y))
print(X.shape)
print(y.shape)

for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print('train data:\n', X[train_index, :])
    print(f"  Test:  index={test_index}")
    print('test data:\n', y[test_index])

print('neg indexing')
li = [0,1,2,3,4,5]
print(li[1:-3])