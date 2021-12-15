import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sklearn

mat = scipy.io.loadmat('data/Indian_pines.mat')
pines = np.array(mat["indian_pines"])
print(pines.shape)
sample = pines[:,:,100]
# plt.imshow(sample)
# plt.show()
print(pines.max())
print(pines.min())


mat_gt = scipy.io.loadmat('data/Indian_pines_gt.mat')
pines_gt = np.array(mat_gt["indian_pines_gt"])
print(pines_gt.shape)
# plt.imshow(sample)
# plt.show()
print(pines_gt.max())
print(pines_gt.min())

mat_corrected = scipy.io.loadmat('data/Indian_pines_corrected.mat')
pines_corrected = np.array(mat_corrected["indian_pines_corrected"])
print(pines_corrected.shape)
# plt.imshow(sample)
# plt.show()
print(pines_corrected.max())
print(pines_corrected.min())

# for i in range(16):
#     print(f"{i} {(pines_gt == i).sum()}")

print("End")
exit(0)

for i in range(220):
    if pines[:,:,i].sum() == 0:
        print(i)