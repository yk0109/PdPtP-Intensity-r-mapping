%load_ext autoreload
%autoreload 2
import AutoDisk as AD
from AutoDisk.autodisk import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cv2
from PIL import Image

folder='E:/PtPdP-Amorphous-Hu'
out = 'analysis-radius-A12'
data_name = f'{folder}/map247-180.dat'
image = Image.open(f'{folder}/map.tif')
X, Y = image.size
print(X,Y)
data_ori = np.memmap(data_name, dtype=np.float32, mode='r', shape=(X*Y, 256, 256))

'(0)Calculate amorphous Carbon backing'
test_position=[]
x0, y0 = 138,9
x1, y1 = 138,48
position = AD.autodisk.bresenham_area(X, Y, x0, y0, x1, y1, y1-y0+1)
#print(position.shape)
for row in position:
    for value in row:
        test_position.append(data_ori[value])
test_position = np.array(test_position).reshape(len(position),len(position),256,256)
avg_pat_test = generateAvg(test_position)
AD.autodisk.saveData(f'{folder}/{out}/C_amorphous.dat',avg_pat_test,dataformat='np.float32')

'(2)Move the mass-center to the Geometry Center and generate profile'
value1=6742
com = ndimage.center_of_mass(data_ori[value1])
com = (int(round(com[0])), int(round(com[1])))
print(com)
com = ndimage.center_of_mass(avg_pat_test)
com = (int(round(com[0])), int(round(com[1])))
print(com)
data_0,center_0 = AD.autodisk.move_center(data_ori[value1], com, square_size=200, plot=True)
data_1,center_1 = AD.autodisk.move_center(avg_pat_test, com, square_size=200, plot=True)
'''
inner_radius=48
outer_radius=49
mask = np.zeros(data_0.shape, dtype=bool)
for i in range(data_0.shape[0]):
    for j in range(data_0.shape[1]):
        distance = np.sqrt((i - com[0])**2 + (j - com[1])**2)
        if inner_radius <= distance <= outer_radius:
            mask[i, j] = True
ring_sum0 = np.sum(data_0[mask])
ring_sum1 = np.sum(data_1[mask])
print(ring_sum0,ring_sum1)
'''
avg_pat_noneback_s = data_0-data_1
avg_pat_noneback_s[avg_pat_noneback_s < 0] = 0
AD.autodisk.visual(np.sqrt(np.sqrt(avg_pat_noneback_s)))

distances, intensities = AD.autodisk.radial_intensity(avg_pat_noneback_s, bin_width=1)
arr = [distances, intensities]
transposed_arr = [list(row) for row in zip(*arr)]
AD.autodisk.saveData(f'{folder}/{out}/profile/profile-data.dat', transposed_arr, overwrite=True)
plt.figure(figsize=(10, 5))
plt.plot(distances[:80], intensities[:80])
plt.xlabel('Distance from center (pixels)')
plt.ylabel('Integrated Intensity')
plt.title('Radial Intensity Profile')
plt.grid(True)
plt.savefig(f'{folder}/{out}/profile/profile-data.png', dpi=300)
plt.show()

popt = AD.autodisk.gauss_fit(arr, num_peaks=2, peak_ranges = [(25, 45), (55, 70)], plot=True, 
                             outdir=f'{folder}/{out}/profile/fit-{value1}.png')

print(popt)
'''
data_new,center_new = AD.autodisk.move_center(avg_pat_noneback_s, com, square_size=150, plot=True)
distances, intensities = AD.autodisk.radial_intensity(data_new, bin_width=1)
arr = [distances, intensities]
transposed_arr = [list(row) for row in zip(*arr)]
#AD.autodisk.saveData(f'{folder}/{out}/profile/profile-{value1}.dat', transposed_arr)
popt = AD.autodisk.gauss_fit(arr, num_peaks=2, peak_ranges = [(25, 45), (55, 70)], plot=True, 
                             outdir=f'{folder}/{out}/profile/fit-{value1}.png')
results.append(popt[1])
print(popt[1])'''