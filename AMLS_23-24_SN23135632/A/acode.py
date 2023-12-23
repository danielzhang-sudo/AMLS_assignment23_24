import numpy as np
import matplotlib.pyplot as plt

def main():
    pneumonia = np.load('Datasets/pneumoniamnist.npz')
    X_train = pneumonia['train_images']
    plt.imshow(X)

    return
