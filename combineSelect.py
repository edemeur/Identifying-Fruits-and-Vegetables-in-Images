import cv2
import os
import glob

# Combines a choice of the first four datasets
# Paths may need to be editted

original = False
circle = True
half = True
quater = False


train_dirs = []
valid_dirs = []
test_dirs = []

comb_name = ""

if original:
    train_dirs.append("archive\\combined_dataset\\Train")
    valid_dirs.append("archive\\combined_dataset\\Valid")
    test_dirs.append("archive\\combined_dataset\\Test")
    comb_name = comb_name + "o"
if circle:
    train_dirs.append("archive\\combined_circle\\Train")
    valid_dirs.append("archive\\combined_circle\\Valid")
    test_dirs.append("archive\\combined_circle\\Test")
    comb_name = comb_name + "c"
if half:
    train_dirs.append("archive\\combined_half\\Train")
    valid_dirs.append("archive\\combined_half\\Valid")
    test_dirs.append("archive\\combined_half\\Test")
    comb_name = comb_name + "h"
if quater:
    train_dirs.append("archive\\combined_quater\\Train")
    valid_dirs.append("archive\\combined_quater\\Valid")
    test_dirs.append("archive\\combined_quater\\Test")
    comb_name = comb_name + "q"

names = []
for name in glob.glob(train_dirs[0] + "/*"):
    names.append(name[81:])
print(names)

new_dir = "archive/combined_{}/".format(comb_name)
sub_folders = ["Test", "Valid", "Train"]
os.mkdir(new_dir)
for x in sub_folders:
    os.mkdir(new_dir+x+"/")
    for name in names:
        dir = new_dir+x+"/" + name
        os.mkdir(dir)

train_images = []
valid_images = []
test_images = []

for x in names:
    train_temp = []
    valid_temp = []
    test_temp = []
    for i in range(len(train_dirs)):
        for image in glob.glob(train_dirs[i] + "/" + x + "/*"):
            train_temp.append(cv2.imread(image))
        for image in glob.glob(valid_dirs[i] + "/" + x + "/*"):
            valid_temp.append(cv2.imread(image))
        for image in glob.glob(test_dirs[i] + "/" + x + "/*"):
            test_temp.append(cv2.imread(image))
    train_images.append(train_temp)
    valid_images.append(valid_temp)
    test_images.append(test_temp)

for count, name in enumerate(names):
    imagecount = 0
    dir = "archive/combined_{}/Train/".format(comb_name) + name
    for image in train_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1
    imagecount = 0
    dir = "archive/combined_{}/Valid/".format(comb_name) + name
    for image in valid_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1
    imagecount = 0
    dir = "archive/combined_{}/Test/".format(comb_name) + name
    for image in test_images[count]:
        filename = f"/image_{imagecount:04}.png"
        cv2.imwrite(dir + filename, image)
        imagecount += 1