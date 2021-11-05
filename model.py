from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard


def create_model():
    model=Sequential()
    model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(10,126)))
    model.add(LSTM(128,return_sequences=True,activation='relu'))
    model.add(LSTM(64,return_sequences=False,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(26,activation='softmax'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    return model