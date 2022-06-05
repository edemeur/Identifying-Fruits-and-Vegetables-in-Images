import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import Counter
from scipy.spatial.distance import cdist
import math
import argparse

# FindPeak without optimisations
def findpeak(data, idx, r):
    new_mean = copy.deepcopy(data[idx])
    done = False
    while not done:
        mean = copy.deepcopy(new_mean)
        new_mean = np.zeros(len(mean))
        distances = cdistance(mean, data)
        found = np.argwhere(distances <= r)
        for y in range(len(mean)):
            new_mean[y] = data[found[:,0], y].mean()
        done = find_threshold(mean, new_mean)
    peak = copy.deepcopy(new_mean)
    return peak

#Mean Shift without optimisations
def meanshift(data, r):
    labels = np.zeros(len(data))
    peaks = copy.deepcopy(data)
    max_label = 1
    for x in range(len(data)):
        peaks[x] = findpeak(data, x, r)
        no_range = True
        for i in range(1, max_label):
            labelled = np.argwhere(labels == i)
            y = labelled[0][0]
            if euclid(peaks[y], peaks[x]) <= r/2:
                labels[x] = labels[y]
                peaks[x] = peaks[y]
                no_range = False
                break
        if no_range:
            labels[x] = max_label
            max_label += 1
    print(np.unique(labels))
    print(Counter(labels))
    return labels, peaks

# FindPeak with optimisations
# Stores points with r/c of the path of the mean, and sends them with the peak
def find_peak_opt(data, idx, r, threshold=0.01, c=4):
    new_mean = copy.deepcopy(data[idx])
    cpts = np.zeros(len(data))
    done = False
    while not done:
        mean = copy.deepcopy(new_mean)
        new_mean = np.zeros(len(mean))
        distances = cdistance(mean, data)
        cpts_found = np.argwhere(distances <= r/c)
        if len(cpts_found) != 0:
            cpts[cpts_found[0]] = 1
        found = np.argwhere(distances <= r)
        for y in range(len(mean)):
            new_mean[y] = data[found[:,0], y].mean()
        done = find_threshold(mean, new_mean, threshold)
    peak = copy.deepcopy(new_mean)
    return peak, cpts

#Mean Shift with optimisations
#Needs to use find_peak_opt to receive cpts points
#Used optimization where all points within r of the peak are also assignmed that peak
def meanshift_opt(data, r, c=4):
    labels = np.zeros(len(data))
    peaks = np.zeros((len(data), len(data[0])))
    max_label = 1
    for x in range(len(data)):
        # if x%100==0:
        #     print(x)
        if labels[x] > 0.0:
            continue
        peaks[x], cpts = find_peak_opt(data, x, r, c)
        no_range = True
        distances = cdistance(peaks[x], data)
        found = np.argwhere(distances <= r)
        cpts_found = np.argwhere(cpts > 0)
        for i in range(1,max_label):
            labelled = np.argwhere(labels == i)
            if len(labelled) != 0:
                y = labelled[0][0]
            else:
                y = 0
            if euclid(peaks[y], peaks[x]) <= r/2:
                labels[x] = labels[y]
                peaks[x] = peaks[x]
                labels[found[:,0]] = labels[y]
                peaks[found[:,0]] = peaks[y]
                labels[cpts_found[:,0]] = labels[y]
                peaks[cpts_found[:,0]] = peaks[y]
                no_range = False
                break
        if no_range:
            labels[x] = max_label
            labels[found[:,0]] = labels[x]
            peaks[found[:,0]] = peaks[x]
            labels[cpts_found[:,0]] = labels[x]
            peaks[cpts_found[:,0]] = peaks[x]
            max_label += 1
    print(np.unique(labels))
    print(Counter(labels))
    return labels, peaks

#Main method that starts segmIm
def segmIm(im, r, positional=False, opt=True):
    data = create_data(im, positional)
    if opt:
        labels, peaks = meanshift_opt(data, r, 4)
    else:
        labels, peaks = meanshift(data, r)
    label_count = 0
    seg_im = copy.deepcopy(im)
    for x in range(len(im)):
        for y in range(len(im[x])):
            if positional:
                seg_im[x][y] = peaks[label_count][:3]
            else:
                seg_im[x][y] = peaks[label_count]
            label_count +=1
    return seg_im

#Method used to quickly show an image
def show_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Transforms img to usable data
# Positional allows for 5D with positions
def create_data(img, positional=False):
    data = np.array(img[0, :, :])
    for x in range(1, img.shape[0]):
        data = np.row_stack([data, img[x, :, :]])
    data = data.astype(np.float64)
    if positional:
        new_data = np.zeros([len(data), 5])
        count = 0
        for x in range(len(img)):
            for y in range(len(img[x])):
                new_data[count] = np.concatenate([data[count], [x * (255/len(img)), y * (255/len(img[0]))]])
                count += 1
        return new_data
    return data

# Method used to get distance between point and data
# Uses cdist from scipy
def cdistance(point, data):
    point = np.array([point])
    distances = cdist(data, point, 'euclidean')
    return distances

# Get euclidean distance between two points
def euclid(point1, point2):
    distance = 0
    for y in range(len(point1)):
        distance += float(point1[y] - point2[y])**2
    return math.sqrt(distance)

# Used to check if means are equal enough within threshold
def find_threshold(mean, new_mean, threshold=0.01):
    done = True
    for x in range(len(mean)):
        if not (abs(mean[x] - new_mean[x]) < threshold):
            done = False
            break
    return done

parser = argparse.ArgumentParser()
parser.add_argument('-image', type=str, dest='image')
parser.add_argument('-radius', dest='r', type=int, default=100)
parser.add_argument('-feature_type',  dest='feature_type', type=int)

args = parser.parse_args()
if args.feature_type == 0:
    FiveD = True
else:
    FiveD = False

min_length = 200
img = cv2.imread(args.image)
width, height, features = img.shape
img = cv2.resize(img, (int(height/width * min_length), min_length))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
segIm = segmIm(img_lab, args.r, FiveD)
show_img(segIm)
segIm_RGB = cv2.cvtColor(segIm, cv2.COLOR_LAB2RGB)
show_img(segIm_RGB)
