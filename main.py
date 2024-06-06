import numpy as np
import random


table_list = [[random.random() for col in range(3)] for row in range(100)]
# print(table_list)

arr = np.random.rand(100, 3)
print(arr.shape)
print(arr[0, 1])


print(table_list[1:4])
print(arr[1:4, 1:arr.shape[1]+1])

new_arr = arr.flatten()
new_arr.reshape(30, 10)
print(new_arr)

print(np.sum(new_arr))
print(new_arr.sum())
print(2+new_arr)
arr = arr + [1, 2, 3]
print(arr)
print(arr > 2)

print(type(arr))
print(list(arr))