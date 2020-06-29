from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall
from tensorflow.keras.optimizers import RMSprop

input_shape=(256,256,3)

def get_model():

    model=Sequential()

    model.add(Conv2D(64,(2,2),input_shape=input_shape,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(256,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Conv2D(512,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())


    model.add(Dropout(0.4))


    model.add(Flatten())


    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.4))


    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model