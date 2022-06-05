import copy
import cv2
import os
import glob

# Creates the Circle dataset
# May require alterations to the paths for folders
# All folders in paths must be created by used

def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


train_dir = "archive/combined_dataset/Train/"
valid_dir = "archive/combined_dataset/Valid/"
test_dir = "archive/combined_dataset/Test/"

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

for i in range(3):
    if i == 0:
        temp_dir = "archive/combined_circle/Train/"
    elif i == 1:
        temp_dir = "archive/combined_circle/Valid/"
    else:
        temp_dir = "archive/combined_circle/Test/"
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
            for r in range(5, 25, 5):
                new_image = copy.deepcopy(image)
                for x in range(0, 100):
                    for y in range(0, 100):
                        if (x - 50) ** 2 + (y - 50) ** 2 <= r ** 2:
                            new_image[x][y] = [255, 255, 255]
                temp_set.append(new_image)
        all_obs_images[idx].append(temp_set)

for count, name in enumerate(names):
    imagecount = 0
    dir = "archive/combined_circle/Train/" + name
    for image in train_obs_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1
    imagecount = 0
    dir = "archive/combined_circle/Valid/" + name
    for image in valid_obs_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1
    imagecount = 0
    dir = "archive/combined_circle/Test/" + name
    for image in test_obs_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1

