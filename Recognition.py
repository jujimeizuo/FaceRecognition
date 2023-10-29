import numpy as np
import cv2

def recognition(TestImage, m, A, Eigenfaces):
    # Projecting centered image vectors into facespace
    # All centered images are projected into facespace by multiplying in
    # Eigenface basis's. Projected vector of each face will be its corresponding
    # feature vector.

    ProjectedImage = np.dot(Eigenfaces.T, A)

    # Extracting the PCA features from test image
    InputImage = cv2.imread(TestImage, cv2.IMREAD_GRAYSCALE)
    temp = np.array(InputImage)

    InImage = temp.reshape(-1) # Vectorizing the image
    Difference = InImage.astype(float) - m # Centered test image
    ProjectedTestImage = np.dot(Eigenfaces.T, Difference) # Test image feature vector
    ProjectedTestImage = ProjectedTestImage.reshape(-1,1)

    # Calculating Euclidean distances
    # Euclidean distances between the projected test image and the projection
    # of all centered training images are calculated. Test image is
    # supposed to have minimum distance with its corresponding image in the
    # training database.

    Euc_dist = np.linalg.norm(ProjectedTestImage - ProjectedImage, axis=0)**2

    Recognized_index = np.argmin(Euc_dist)
    OutputName = str(Recognized_index+1) + '.jpg'

    return OutputName