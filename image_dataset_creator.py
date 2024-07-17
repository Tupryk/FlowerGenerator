import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIMS = 64


def prepare_dataset():
    data = []
    paths = ["./data/flowers/daisy", "./data/flowers/dandelion",
             "./data/flowers/rose", "./data/flowers/sunflower", "./data/flowers/tulip"]
    for path in paths:
        for im_file in os.listdir(path):
            image = cv2.imread(f"{path}/{im_file}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_DIMS, IMAGE_DIMS))
            image = image/255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            data.append(image)

    data = np.array(data)
    return data


data = prepare_dataset()
np.savez("./data/arrays.npz", data)
data = np.load("./data/arrays.npz")["arr_0"]

print(data.shape)
image = data[np.random.randint(0, len(data))]
image = np.transpose(image, (1, 2, 0))
plt.imshow(image)
plt.show()
