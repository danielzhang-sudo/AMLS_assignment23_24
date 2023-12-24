import numpy as np
from helper import helper

def main(args):
    task = args.task
    mode = args.mode
    alg = args.alg
    ckpt_path = args.ckpt_path

    # Open file
    pneumonia = np.load('./AMLS_23-24_SN23135632/Datasets/pneumoniamnist.npz')
    
    # Extract training, validation, test images
    x_train = pneumonia['train_images']
    x_val = pneumonia['val_images']
    x_test = pneumonia['test_images']

    y_train = pneumonia['train_labels']
    y_val = pneumonia['val_labels']
    y_test = pneumonia['test_labels']

    # Reshape the 2d image into a 1d image    
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
    X_val = x_val.reshape(x_val.shape[0], x_val.shape[1]**2)

    classes = ['normal', 'pneumonia']

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

