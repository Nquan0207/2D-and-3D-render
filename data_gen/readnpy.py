import numpy as np

mask = np.load("/Users/quan0207/School/computer graphics/simpleviewer/data_gen/output/mask/000000.npy")
print(mask.shape, mask.dtype)
print(np.unique(mask))  # xem các instance_id
