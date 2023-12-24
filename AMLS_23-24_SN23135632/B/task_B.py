import numpy as np
from helper import helper
from skimage.color import rgb2gray

def main(args):
    task = args.task
    mode = args.mode
    alg = args.alg
    ckpt_path = args.ckpt_path

    # Open file
    path = np.load('./Datasets/pathmnist.npz')
    
    # Extract training, validation, test images
    x_train = path['train_images']
    x_val = path['val_images']
    x_test = path['test_images']

    y_train = path['train_labels']
    y_val = path['val_labels']
    y_test = path['test_labels']

    # Change RGB image to grayscale image
    x_train = rgb2gray(x_train)
    x_val = rgb2gray(x_val)
    x_test = rgb2gray(x_test)

    # Reshape the 2d image into a 1d image    
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
    X_val = x_val.reshape(x_val.shape[0], x_val.shape[1]**2)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    x = ''
    y = ''

    if mode == "training":
        x = X_train
        y = y_train
    elif mode == "validation":
        x = X_val
        y = y_val
    elif mode == "Testing":
        x = X_test
        y = y_test

    helper(task, mode, alg, ckpt_path, x, y, classes)