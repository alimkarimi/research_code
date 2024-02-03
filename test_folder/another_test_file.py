def final_test():
    print('found this too!!!')

li = [0,1,2,3,5]
print(li[1:-1])

#standard scaling
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np

# 4 samples/observations and 2 variables/features
data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
mean1 = data[:,0].mean()
mean2 = data[:,1].mean()
std1 = data[:,0].std()
std2 = data[:,1].std()
print(mean1, mean2, std1, std2)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(data.shape)
print(data)
print(scaled_data)