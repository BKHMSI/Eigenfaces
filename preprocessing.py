import cv2
import matplotlib.pyplot as plt 

def histogram_eq(img):
    return cv2.equalizeHist(img)

def facechop(img):  
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w]
        return sub_face

def test(img_path):
    img = plt.imread(img_path)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = facechop(img)
    img = histogram_eq(img)
    plt.imshow(img, cmap='gray')
    plt.show()

# test('images/uknown.jpg')