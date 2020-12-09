from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def gaussian_curvature(Z):
    # usage: 
    # curvature = gaussian_curvature(image_gray)
    # norm_image = cv2.normalize(curvature, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)

    # Zy, Zx = np.gradient(Z)                                                     
    # Zxy, Zxx = np.gradient(Zx)                                                  
    # Zyy, _ = np.gradient(Zy)                                                    
    # K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2             

    dx, dy = np.gradient(Z)
    dxx, dxy = np.gradient(dx)
    dyx, dyy = np.gradient(dy)

    # det of hessian
    ret = dxx*dyy - dxy*dxy
    ret /= (1 + dx*dx + dy*dy) * (1 + dx*dx + dy*dy)

    return ret

# image = data.hubble_deep_field()[0:500, 0:500]
image = mpimg.imread('1590993664526.png')
image = np.array(image, dtype=np.double)
image_gray = rgb2gray(image)
max_sigma = 10 # default: 30
blobs_log = blob_log(image_gray, max_sigma=max_sigma, num_sigma=10, threshold=0.07)

blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=max_sigma, threshold=0.07)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=max_sigma, threshold=0.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()


dx = [-1, 0, 1, 0, -1, -1, 1, 1]
dy = [0, 1, 0, -1, -1, 1, -1, 1]


for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    total = len(blobs)
    small_radius = 0
    removed = 0

    copy = image.copy()


    for blob in blobs:
        y, x, r = blob
        r = int(r)
        y = int(y)
        x = int(x)

        maxValCounts = []

        #for too samll blob

        if r <= 1:
            small_radius += 1
            continue

        # if r < y-r and y+r < image_gray.shape[0]-r and r < x-r and x+r < image_gray.shape[1]-r:
        if r+4 < y-r and y+r < image_gray.shape[0]-r-4 and r+4 < x-r and x+r < image_gray.shape[1]-r-4:
            # cv2.circle(image_gray, (x, y), r, (255, 0, 0), -1)

            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(mask, (x, y), r, (1.0), -1)
            mask = mask[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring1, (x, y), r+1, (1.0), 1)
            ring1 = ring1[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring2, (x, y), r+2, (1.0), 1)
            ring2 = ring2[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring3 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring3, (x, y), r+3, (1.0), 1)
            ring3 = ring3[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            ring4 = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
            cv2.circle(ring4, (x, y), r+3, (1.0), 1)
            ring4 = ring4[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]

            out = image_gray[y-2*r-4:y+2*r+4, x-2*r-4:x+2*r+4]
            # out = image_gray[y-r:y+r, x-r:x+r]
            
            inner = out[np.where(mask == 1.0)]
            outer1 = out[np.where(ring1 == 1.0)]
            outer2 = out[np.where(ring2 == 1.0)]
            outer3 = out[np.where(ring3 == 1.0)]
            outer4 = out[np.where(ring4 == 1.0)]

            innerAvg = sum(inner) / len(inner)
            avg1 = sum(outer1) / len(outer1)
            avg2 = sum(outer2) / len(outer2)
            avg3 = sum(outer3) / len(outer3)
            avg4 = sum(outer4) / len(outer4)
            threshold = innerAvg * 3 / 7

            # print('r: ', r)
            # print('thresh: ', threshold)
            # print(innerAvg)
            # print(avg1)
            # print(avg2)
            # print(avg3)
            # print('')

            if avg1 < threshold or avg2 < threshold or avg3 < threshold or avg4 < threshold:
            # if avg1 < threshold or avg2 < threshold or avg3 < threshold:
                # true void
                
                c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
                ax[idx].add_patch(c)

                print('r: ', r)
                print('thresh: ', threshold)
                print(innerAvg)
                print(avg1)
                print(avg2)
                print(avg3)
                print('')
                
                cv2.circle(copy, (x, y), r, (0, 1.0, 0), 1)

                cv2.imshow('out', out)
                cv2.imshow('ret', copy)

                while True:
                    ch = cv2.waitKey(1)
                    if ch == 27:
                        break
            else:

                cv2.circle(copy, (x, y), r, (0, 0, 1.0), 1)

                c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
                ax[idx].add_patch(c)
                removed += 1

            # c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
            # ax[idx].add_patch(c)

    # cv2.imshow('ret', copy)

    # while True:
    #     ch = cv2.waitKey(1)
    #     if ch == 27:
    #         break

            
                

    ax[idx].set_axis_off()
    print(total, small_radius, removed)
    print('total: ', total)
    print('small R: ', small_radius)
    print('removed: ', removed, removed / (total - small_radius))


plt.tight_layout()
plt.show()

# cv2.imshow('ret', image)
# while True:
#     ch = cv2.waitKey(1)
#     if ch == 27:
#         break


