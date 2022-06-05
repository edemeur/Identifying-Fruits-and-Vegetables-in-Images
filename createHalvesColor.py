import copy
import cv2
import os
import glob
import sys

# Creates the Half color dataset
# May require alterations to the paths for folders
# All folders in paths must be created by used


train_dir = "archive/combined_dataset/Train/"
valid_dir = "archive/combined_dataset/Valid/"
test_dir = "archive/combined_dataset/Test/"
color = "forest" # forest, red, green, blue

names = []
for name in glob.glob(train_dir + "*"):
    names.append(name[83:])
print(names)
train_images = []
valid_images = []
test_images = []

for x in names:
    train_temp = []
    valid_temp = []
    test_temp = []
    for image in glob.glob(train_dir + x + "/*"):
        train_temp.append(cv2.imread(image))
    for image in glob.glob(valid_dir + x + "/*"):
        valid_temp.append(cv2.imread(image))
    for image in glob.glob(test_dir + x + "/*"):
        test_temp.append(cv2.imread(image))
    train_images.append(train_temp)
    valid_images.append(valid_temp)
    test_images.append(test_temp)


color = "forest"

if color.lower() == "green":
    color_values = [0,255,0]
elif color.lower() == "red":
    color_values = [255,0,0]
elif color.lower() == "blue":
    color_values = [0 ,0, 255]
elif color.lower() == "forest":
    color_values = [34, 139, 34]
else:
    sys.exit("No such color exists")

for i in range(3):
    if i == 0:
        temp_dir = "archive/combined_half_{}/Train/".format(color)
    elif i == 1:
        temp_dir = "archive/combined_half_{}/Valid/".format(color)
    else:
        temp_dir = "archive/combined_half_{}/Test/".format(color)
    for name in names:
        dir = temp_dir + name
        os.mkdir(dir)

train_obs_images = []
valid_obs_images = []
test_obs_images = []

all_images = [train_images, valid_images, test_images]
all_obs_images = [train_obs_images, valid_obs_images, test_obs_images]


for idx, superset in enumerate(all_images):
    for set in superset:
        temp_set = []
        for image in set:
            for i in range(0, 4):
                new_image = copy.deepcopy(image)
                if i == 0:
                    for x in range(0, 50):
                        for y in range(0, 100):
                            new_image[x][y] = color_values
                if i == 1:
                    for x in range(0, 100):
                        for y in range(0, 50):
                            new_image[x][y] = color_values
                if i == 2:
                    for x in range(50, 100):
                        for y in range(0, 100):
                            new_image[x][y] = color_values
                if i == 3:
                    for x in range(0, 100):
                        for y in range(50, 100):
                            new_image[x][y] = color_values
                temp_set.append(new_image)
        all_obs_images[idx].append(temp_set)

for count, name in enumerate(names):
    imagecount = 0
    dir = "archive/combined_half_{}/Train/".format(color) + name
    for image in train_obs_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1
    imagecount = 0
    dir = "archive/combined_half_{}/Valid/".format(color) + name
    for image in valid_obs_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1
    imagecount = 0
    dir = "archive/combined_half_{}/Test/".format(color) + name
    for image in test_obs_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1

