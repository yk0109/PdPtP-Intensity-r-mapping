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
#AD.autodisk.visual(np.sqrt(np.sqrt(avg_pat_test)),outdir=f'{folder}/{out}/C_amorphous.png')

'(1)Calculate the center-mass and Offset amorphous Carbon backing'
all_position = []
x2, y2 = 63, 18
x3, y3 = 63, 41
area = AD.autodisk.bresenham_area(X, Y, x2, y2, x3, y3, y3-y2+1)
AD.autodisk.saveData(f'{folder}/{out}/area.dat', area, overwrite=True)
# Visualization
data_map = AD.autodisk.readDatadat(f'{folder}/map.dat',dim=1,pixel=180,pixel2=247)
mask_area = np.zeros_like(data_map, dtype=bool)
mask_area[y2:y3, x2:(x2+y3-y2)] = True
fig, ax = plt.subplots()
ax.imshow(data_map, cmap='gray')
rect = patches.Rectangle((x2, y2), y3-y2-1, y3-y2-1, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.title('Data Map with Calculation Area')
plt.axis('off')
plt.savefig(f'{folder}/{out}/area.png', dpi=300)
plt.show()

for row in area:
    for value in row:
        all_position.append(data_ori[value])
all_position = np.array(all_position).reshape(len(area),len(area),256,256)
avg_pat_all = AD.autodisk.generateAvg(all_position)

com = ndimage.center_of_mass(avg_pat_all)  # Calculate the center of mass within the specified range
com = (int(round(com[0])), int(round(com[1])))
print(com)
AD.autodisk.visual(np.sqrt(np.sqrt(avg_pat_all)),outdir=f'{folder}/{out}/avg_pat_all.png',point=[com,2])

avg_pat_noneback = AD.autodisk.generateAvg(all_position)-avg_pat_test
threshold_value = 0
avg_pat_noneback[avg_pat_noneback < threshold_value] = 0
AD.autodisk.saveData(f'{folder}/{out}/avg_pat_noneback.dat',avg_pat_noneback,dataformat='np.float32')
AD.autodisk.visual(np.sqrt(np.sqrt(avg_pat_noneback)),outdir=f'{folder}/{out}/avg_pat_noneback.png',point=[com,2])

'(2)Move the mass-center to the Geometry Center and generate profile'
results=[]
for row1 in area:
    for value1 in row1:
        #value1=6499
        avg_pat_noneback_s = data_ori[value1]-avg_pat_test
        avg_pat_noneback_s[avg_pat_noneback_s < 0] = 0
        data_new,center_new = AD.autodisk.move_center(avg_pat_noneback_s, com, square_size=150, plot=True)
        distances, intensities = AD.autodisk.radial_intensity(data_new, bin_width=1)
        arr = [distances, intensities]
        transposed_arr = [list(row) for row in zip(*arr)]
        #AD.autodisk.saveData(f'{folder}/{out}/profile/profile-{value1}.dat', transposed_arr)
        popt = AD.autodisk.gauss_fit(arr, num_peaks=2, peak_ranges = [(25, 45), (55, 70)], plot=True, 
                                     outdir=f'{folder}/{out}/profile/fit-{value1}.png')
        results.append(popt[1])
        print(popt[1])
results = np.array(results).reshape(len(area),len(area))
AD.autodisk.saveData(f'{folder}/{out}/radius.dat',results)