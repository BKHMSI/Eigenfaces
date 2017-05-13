from __future__ import print_function
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os 
from scipy.misc import imresize
from preprocessing import facechop, histogram_eq


DATA_PATH = '/Users/BAlKhamissi/Documents/Datasets/yalefaces'
num_faces = 15
num_per_face = 11
num_images = num_faces * num_per_face
# original images 320 * 243
m = 100
n = 100
img_dim = (m,n)
input_dim = (num_images, img_dim[0], img_dim[1])
X_train = np.zeros(input_dim, dtype='uint8')
X_val = np.zeros(input_dim, dtype='uint8')
X_mean = np.zeros(img_dim, dtype='uint8')
Y_train = np.zeros(num_images)

EPSIOLON_ID = 3000
EPSIOLON_DETECT = 2400

imgx = cv.imread('images/unknown.jpg', 0)
print(imgx)
gg = input("foo")

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

def get_noise():
    return np.random.rand(m,n)

def get_unknown_face():
    img = plt.imread('images/badr.jpg')
    #img = plt.imread('images/unknown.jpg')
    return img

def get_building():
    img = cv.imread('images/building.jpg', 0)
    return img

# Step 1
def read_data():
    t_idx = v_idx = 0
    for face in range(num_faces):
        face_path = os.path.join(DATA_PATH, str(face+1).zfill(2))
        for i, file in enumerate(os.listdir(face_path)):
            file_path = os.path.join(face_path, file)
            img = plt.imread(file_path, 0)
            img = facechop(img)
            img = histogram_eq(img)
            if(i <= 9):
                X_train[t_idx] = imresize(img, img_dim)
                Y_train[t_idx] = face
                t_idx = t_idx + 1
            else: 
                X_val[v_idx] = imresize(img, img_dim)
                v_idx = v_idx + 1
    return X_train, Y_train, X_val

def train(X_train):
    # Unroll Images
    X_train = X_train.reshape(X_train.shape[0], -1).T

    # Step 2
    X_mean = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)

    A = X_train - X_mean 

    # Step 3
    U, S, V = np.linalg.svd(A)
    rank = np.sum(S > 1e-12)

    # Step 4
    Ur = U[:, :rank]
    X = Ur.T.dot(A)
    return X, Ur, X_mean

def predict(img, X, Ur, X_mean, Y_train):

    img = img.reshape(1, -1).T
    a = img - X_mean
    x = Ur.T.dot(a)

    # detect face
    ap = Ur.dot(x)
    ep_a_x = np.linalg.norm(a - ap)
    print("e_f:", ep_a_x)
    if ep_a_x > EPSIOLON_DETECT:
        print("Not a face.")
        return -2
    # end detect face

    # What is this part?
    D = X - x*np.ones((1,X.shape[0]))
    d = np.sqrt(np.diag(D.T.dot(D)))
    d_idx = np.argmin(d)
    print("Distance: ", d[d_idx])
    if d[d_idx] < EPSIOLON_ID:
        print(f"Face detected and IDed as #{Y_train[d_idx]}")
        return d_idx
    else:
        print("Uknown Face")
        return -1

def preprocess(img):
    img = facechop(img)
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = histogram_eq(img)
    img = imresize(img, img_dim)
    return img


X_train, Y_train, X_val = read_data()

X, Ur, X_mean = train(X_train)

# test = X_train[56] # face
# test = get_noise() 
# test = get_unknown_face()
# test = get_building()
try:
    #test = preprocess(test)
    pred = predict(X_val[i], X, Ur, X_mean, Y_train)
except:
    pred = -2

if pred not in {-1, 0}:
    test = test.reshape(img_dim)
    img = X_train[pred].reshape(img_dim)
    subplot(test,img)
elif pred == -1:
    imshow(test, title="Unknown Face")
else:
    imshow(test, title="Not a Face")

# Show Results
X_mean = X_mean.reshape(img_dim)
imshow(X_mean, title='Mean Image')

