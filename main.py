import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os 
from scipy.misc import imresize


DATA_PATH = '/Users/BAlKhamissi/Documents/Datasets/yalefaces'
num_faces = 15
num_per_face = 11
num_images = 165
input_dim = (num_images, 60, 80)
img_dim = (60,80)
X_train = np.zeros(input_dim, dtype='uint8')
Y_train = np.zeros(input_dim, dtype='uint8')
X_mean = np.zeros(input_dim, dtype='uint8')

EPISOLON_1 = 10
EPISOLON_0 = 5

def imshow(img, gray = True, title=''):
    plt.title(title)
    plt.imshow(img, cmap='gray') if gray else plt.imshow(img)
    plt.show()

# Step 1
def read_data():
    for face in xrange(num_faces):
        face_path = os.path.join(DATA_PATH, str(face+1).zfill(2))
        for i, file in enumerate(os.listdir(face_path)):
            file_path = os.path.join(face_path, file)
            img = plt.imread(file_path, 0)
            img = imresize(img, img_dim)
            idx = face*num_per_face+i
            X_train[idx] = img
            Y_train[idx] = face
    return X_train, Y_train

def train(X_train):
    # Unroll Images
    X_train = X_train.reshape(X_train.shape[0], -1)
    print(X_train.shape)

    # Step 2
    X_mean = np.mean(X_train, axis=0)
    A = X_train - X_mean

    # Step 3
    U, S, V = np.linalg.svd(A.T)
    rank = np.sum(S > 1e-12)
    print(U.shape)

    # Step 4
    Ur = U[:, :rank]
    X = Ur.T.dot(A.T)
    return X, Ur, X_mean

def predict(img, X, Ur, X_mean, Y_train):
    img = img.reshape(-1)
    a = img - X_mean
    x = Ur.T.dot(a.T)
    ef = np.linalg.norm(x - Ur.T.dot(x.T))
    if ef < EPISOLON_1:
        D = X - x*np.ones((1,X_train.shape[0]))
        d = np.sqrt(np.diag(D.T.dot(D)))
        d_min, d_idx = np.min(d)
        if d_min < EPISOLON_0:
            print("This is Face #", Y_train[d_idx])
        else:
            print("Uknown Face")
    else:
        print("Image is not a Face")


X_train, Y_train = read_data()
X, Ur, X_mean = train(X_train)

predict(X_train[0], X, Ur, X_mean, Y_train)

# Show Results
X_mean = X_mean.reshape(img_dim)
imshow(img_mean, title='Mean Image')