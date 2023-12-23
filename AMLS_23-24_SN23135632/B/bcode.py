import numpy as np
import matplotlib.pyplot as plt

def main():

# Open file
    pneumonia = np.load('./AMLS_23-24_SN23135632/Datasets/pathmnist.npz')
    
    # Extract training, validation, test images
    x_train = pneumonia['train_images']
    x_val = pneumonia['val_images']
    x_test = pneumonia['test_images']

    y_train = pneumonia['train_labels']
    y_val = pneumonia['val_labels']
    y_test = pneumonia['test_labels']

    return