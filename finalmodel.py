#mobile NET
import keras
#import hw15_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from visual_callbacks import AccLossPlotter
from keras.applications import MobileNet
import numpy as np
import time
from google.colab import drive

drive.mount('/content/drive')



def main():
    np.random.seed(45)
    nb_class = 2
    width, height = 224,224

 #   sn = hw15_model.SqueezeNet(nb_classes=nb_class, inputs=(height, width, 3))
    #sn = VGG16(weights='imagenet',include_top=False)
    sn=keras.applications.mobilenet.MobileNet(classes=2,weights=None)
    print('Build model')

    sgd = SGD(lr=0.01, decay=0.0002, momentum=0.9, nesterov=True)
    sn.compile(
        optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print(sn.summary())

    # Training
    train_data_dir = '/content/drive/My Drive/Assessment/boxes/train'
    validation_data_dir = '/content/drive/My Drive/Assessment/boxes/test'
    nb_train_samples=120
    nb_validation_samples=70
    nb_epoch=100

    #   Generator
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    #train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(width, height),
            batch_size=4,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(width, height),
            batch_size=4,
            class_mode='categorical')

    # Instantiate AccLossPlotter to visualise training
  #  tbcallback = keras.callbacks.TensorBoard(log_dir = '/home/ubuntu/week1_temp/keras/Graph',histogram_freq=0,write_graph=True,write_images=True)
   # early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
    #checkpoint = ModelCheckpoint(                                         
               #     'weights.{epoch:02d}-{val_loss:.2f}.h5',
     #               monitor='val_loss',                               
      #              verbose=0,                                        
       #             save_best_only=True,                              
        #            save_weights_only=True,                           
         #           mode='min',                                       
          #          period=1)                                         
    start=time.time()
    sn.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples, 
      #      callbacks=[tbcallback, checkpoint],
            verbose=2,
            shuffle=True)
    end=time.time()
    sn.save_weights('/content/drive/My Drive/Assessment/boxes/weights_mobilenet.h5')
   
    print(end-start)
#if _name_ == '_main_':
main()
  #  input('Press ENTER to exit...')