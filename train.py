import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard

from model import create_model

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils


class Data:
    def __init__(self):
        self.DATA_PATH=os.path.join('train-data')
        self.actions=[]
        for i in range(97,123):
            self.actions.append(chr(i))
        self.no_sequences=10
        self.sequence_length=10

        self.label_map={label:num for num, label in enumerate(self.actions)}

    def load_data(self):
        sequences,labels=[],[]
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window=[]
                for frame_num in range(self.sequence_length):
                    res=np.load(os.path.join(self.DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[action])

        X=np.array(sequences)
        y=to_categorical(labels).astype(int)
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=0.05)

    def train(self,model,epochs=300):
        log_dir=os.path.join('Logs')
        tb_callback=TensorBoard(log_dir=log_dir)
        X_val=self.X_train[-50:]
        y_val=self.y_train[-50:]
        X_train=self.X_train[:-50]
        y_train=self.y_train[:-50]
        model.fit(X_train,y_train,epochs=epochs,validation_data=(X_val,y_val), callbacks=[tb_callback])
        return model

        #model.save('action.h5')

# res=model.predict(X_test)

# model.save('action.h5')

# data=Data()
# data.load_data()
# model=create_model()
# data.load_data()
# trained=data.train(model,100)
# res=model.predict(data.X_test)
# print('----------------------------------Output------------------------------------')
# print(data.actions[np.argmax(res[2])])
# print(data.actions[np.argmax(res[2])])