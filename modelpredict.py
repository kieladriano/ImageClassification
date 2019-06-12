#Predict
import keras
from keras.optimizers import SGD
import cv2
import numpy as np

sgd = SGD(lr=0.01, decay=0.0002, momentum=0.9, nesterov=True)
model =keras.applications.mobilenet.MobileNet(classes=2,weights=None)

#load the saved weights
model.load_weights('/content/drive/My Drive/Assessment/boxes/weights_mobilenet.h5')
model.compile(
        optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'],)



#img=cv2.imread("/content/drive/My Drive/Assessment/test/NG/IMG_6785.JPG")

#Change the file path to test new image

img=cv2.imread("/content/drive/My Drive/Assessment/test/OK/IMG_6764.JPG")
#pre processing
img=img/255.

print(img.shape)
img=cv2.resize(img,(224,224),3)
#reduce dimension
img=np.expand_dims(img,axis=0)
print(img.shape)

#predict
result=model.predict(img)

#0= Not Good 1=Good
np.argmax(result)