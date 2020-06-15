from model import get_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import os,sys,argparse

def train(train_path,test_path):
    datagen = ImageDataGenerator(rescale=1 / 255)
    train = datagen.flow_from_directory(train_path,
                                        class_mode='binary')
    test = datagen.flow_from_directory(test_path,
                                       class_mode='binary')

    model=get_model()

    early = EarlyStopping(monitor='accuracy', patience=3, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1, cooldown=0, mode='auto',
                                  min_delta=0.0001, min_lr=0)

    model.fit(train, epochs=20, validation_data=(test), shuffle=True, callbacks=[early, reduce_lr])
    model.save('mask_model.h5')

def main():
    my_parser = argparse.ArgumentParser(description='Mask detection')

    my_parser.add_argument('--train_path',
                           metavar='train_path',
                           type=str,
                           help='path of the directory of training images', required=True)
    my_parser.add_argument('--test_path',
                           metavar='test_path',
                           type=str,
                           help='path of the directory of test images', required=True)

    args = my_parser.parse_args()

    train(args.train_path,args.test_path)

if __name__ == '__main__':
    main()

