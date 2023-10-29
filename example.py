from CreateDatabase import create_database
from EigenfaceCore import eigenface_core
from Recognition import recognition
import cv2

TestImage = input('Enter test image name (a number between 1 to 10):')
TestImage = 'TestDatabase/' + TestImage + '.jpg'
print(TestImage)
im = cv2.imread(TestImage, cv2.IMREAD_GRAYSCALE)

T = create_database('TrainDatabase')
m, A, Eigenfaces = eigenface_core(T)

OutputName = recognition(TestImage, m, A, Eigenfaces)

SelectedImage = 'TrainDatabase/' + OutputName
print(SelectedImage)
SelectedImage = cv2.imread(SelectedImage, cv2.IMREAD_GRAYSCALE)

cv2.imshow('Test Image', im)
cv2.imshow('Equivalent Image', SelectedImage)

cv2.waitKey(0)