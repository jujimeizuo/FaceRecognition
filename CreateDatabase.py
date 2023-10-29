import os
import numpy as np
from PIL import Image
import cv2

def create_database(train_database_path):
    # File management
    TrainFiles = os.listdir(train_database_path)
    Train_Number = 0

    for file in TrainFiles:
        if file not in ['.', '..', 'Thumbs.db']:
            Train_Number += 1 # Number of all images in the training database

    # Construction of 2D matrix from 1D image vectors
    T = []
    for i in range(1, Train_Number + 1):
        # I have chosen the name of each image in databases as a corresponding
        # number. However, it is not mandatory!
        str = os.path.join(train_database_path, f"{i}.jpg")
        img = cv2.imread(str, cv2.IMREAD_GRAYSCALE)

        temp = img.reshape(-1)   # Reshaping 2D images into 1D image vectors
        T.append(temp) # 'T' grows after each turn
    T = np.array(T)
    return T
