import numpy as np
import pandas as pd
import re

# s = "dgjsag"
# l = list(s)
# print(l)

# l = [1, 2, 3, 4]
# print(l[1:8])

# l1 = [[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]]
# l1 = np.asarray(l1)
# print(l1.shape)
# l1 = np.reshape(l1, [-1, 8])
# print(l1)
# print(l1.shape)
# l2 = l1
# l = [l1, l2]
# l = np.asarray(l)
# print(l.shape)
# l = np.concatenate(l, 1)
# print(l.shape)

# l = [1, 2, 3, 4]
# a, b, c, d = l
# print(a, b, c, d)

# weights = np.ones(5)
# print(weights)

string = "ssadsa***dsada*sdad"
string = re.sub("\*+", "*", string)
print(string)
