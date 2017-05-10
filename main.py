from __future__ import print_function
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os 
from scipy.misc import imresize



DATA_PATH = '/Users/BAlKhamissi/Documents/Datasets/yalefaces'
num_faces = 15
num_per_face = 11
num_images = num_faces * num_per_face
img_dim = (60,80)
input_dim = (num_images, img_dim[0], img_dim[1])
X_train = np.zeros(input_dim, dtype='uint8')
Y_train = np.zeros(num_images)
X_mean = np.zeros(img_dim, dtype='uint8')

EPSIOLON_ID = 10
EPSIOLON_DETECT = 2

def subplot(X, Y):
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(Y, cmap='gray')
    a.set_title('Prediction')
    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(X,cmap='gray')
    a.set_title('Ground Truth')
    plt.show()

def imshow(img, gray = True, title=''):
    plt.title(title)
    plt.imshow(img, cmap='gray') if gray else plt.imshow(img)
    plt.show()

# Step 1
def read_data():
    for face in range(num_faces):
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
    # global X_mean, X_train
    # Unroll Images
    X_train = X_train.reshape(X_train.shape[0], -1)
    # assert X_train.shape[0] == num_images
    print("X_train.shape:", X_train.shape)

    # Step 2
    X_mean = np.mean(X_train, axis=0)
    A = X_train - X_mean

    # Step 3
    U, S, V = np.linalg.svd(A.T)
    rank = np.sum(S > 1e-12)
    print("U.shape:", U.shape)

    # Step 4
    Ur = U[:, :rank]
    X = Ur.T.dot(A.T)
    return X, Ur, X_mean

def predict(img, X, Ur, X_mean, Y_train):
    img = img.reshape(-1)
    a = img - X_mean
    x = Ur.T.dot(a.T)
    # Detect face
    ap = Ur.dot(x)
    ep_a_x = np.linalg.norm(a - ap)
    print("e_f:", ep_a_x)
    if ep_a_x > EPSIOLON_DETECT:
        print("Not a face.")
        return 0
    # end detect face
    e_face = np.linalg.norm(x - X, axis=0)
    # e_face.reshape(-1)
    print("e_face.shape:", e_face.shape)

    for i, f in enumerate(e_face):
        print(i, f)

    # xx = x - X
    # e_face = np.sqrt(xx.T.dot(xx))
    idx = np.argmin(e_face)

    print("Y_train.shape:", Y_train.shape)
    print(idx)
    print("Face #:", Y_train[idx])
    return idx

    # if e_face < EPSIOLON_ID:
    #     D = X - x*np.ones((1,X_train.shape[0]))
    #     d = np.sqrt(np.diag(D.T.dot(D)))
    #     d_min, d_idx = np.min(d)
    #     if d_min < EPSIOLON_DETECT:
    #         print("This is Face #", Y_train[d_idx])
    #     else:
    #         print("Uknown Face")
    # else:
    #     print("Image is not a Face")


X_train, Y_train = read_data()
X, Ur, X_mean = train(X_train)

#test = X_train[99] # face
#test = np.random.rand*img_dim) # noise
test = plt.imread('building.jpg') # not face
test = plt.im
test = imresize(test, img_dim)
imshow(test)
pred = predict(test, X, Ur, X_mean, Y_train)

test = test.reshape(img_dim)
img = X_train[pred].reshape(img_dim)

subplot(img,test)

# Show Results
X_mean = X_mean.reshape(img_dim)
imshow(X_mean, title='Mean Image')
