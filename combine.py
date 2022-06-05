import cv2
import os
import random
import glob

# Combines the training and test sets and splits into three random IID sets
# May require alterations to the paths for folders
# All folders in paths must be created by used

random.seed(29)

train_dir = "archive/fruits-360_dataset/fruits-360/Training/"
test_dir = "archive/fruits-360_dataset/fruits-360/Test/"

names = []
for name in glob.glob(train_dir +  "*"):
    names.append(name[99:])

images = []

for x in names:
    temp = []
    for image in glob.glob(train_dir + x + "/*"):
        temp.append(cv2.imread(image))
    for image in glob.glob(test_dir + x + "/*"):
        temp.append(cv2.imread(image))
    images.append(temp)

count = 0
imagecount = 0
for name in names:
    dir = "archive/combined_dataset/All/" + name
    os.mkdir(dir)
    for image in images[count]:
        filename = f"/image_{imagecount:06}.png"
        cv2.imwrite(dir+filename, image)
        imagecount+=1
    count+=1
    imagecount=0

for i in range(3):
    if i == 0:
        temp_dir = "archive/combined_dataset/Train/"
    elif i == 1:
        temp_dir = "archive/combined_dataset/Valid/"
    else:
        temp_dir = "archive/combined_dataset/Test/"
    for name in names:
        dir = temp_dir + name
        os.mkdir(dir)

first = True
for idx, set in enumerate(images):
    random.shuffle(set)
    length = len(set)
    train_len = int(length * 0.7)
    valid_len = int(length * 0.1)
    test_len = length - train_len - valid_len
    if first:
        first = False
        print(train_len, valid_len, test_len)
    for i in range(3):
        if i == 0:
            temp_dir = "archive/combined_dataset/Train/"
            temp_set = set[:train_len]
        elif i == 1:
            temp_dir = "archive/combined_dataset/Valid/"
            temp_set = set[train_len: valid_len+train_len]
        else:
            temp_dir = "archive/combined_dataset/Test/"
            temp_set = set[valid_len+train_len:]
        count = 0
        imagecount = 0
        dir = temp_dir + names[idx]
        for image in temp_set:
            filename = f"/image_{imagecount:06}.png"
            cv2.imwrite(dir + filename, image)
            imagecount += 1
        count += 1
        imagecount = 0