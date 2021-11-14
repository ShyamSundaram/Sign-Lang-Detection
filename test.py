import mediapipe as mp
import cv2
import numpy as np

from data_collection import draw_landmarks, extract_keypoints, mediapipe_detection

from model import create_model

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils

sequence=[]
sentence=[]
thresh=0.8
actions=[]
for i in range(97,123):
    actions.append(chr(i))

cap=cv2.VideoCapture(0)

model=create_model()
model.load_weights('action3.h5')
res=np.array([2.9485551e-20,8.7930697e-18,4.4212179e-16,3.8555774e-24,4.7811854e-32
,3.9088125e-11,6.2194567e-37,4.1775081e-21,1.1696461e-07,9.9999988e-01
,4.5697821e-24,2.0483630e-15,5.7101654e-12,4.6459605e-17,2.9196457e-08
,8.6699751e-23,1.4897640e-13,2.7513793e-18,2.8300720e-15,1.8070643e-18
,1.9550273e-19,4.8556037e-26,2.5306402e-19,3.2431286e-26,2.4559270e-11
,1.7702510e-22])

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame=cap.read()

        image,results=mediapipe_detection(frame,holistic)

        draw_landmarks(image,results)

        keypoints=extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence=sequence[:10]
        #print(keypoints)
        if len(sequence)==10:
            res=model.predict(np.expand_dims(sequence,axis=0))[0]
            print(actions[np.argmax(res)])
        else:
            continue
        print(res[np.argmax(res)])
        if res[np.argmax(res)]>thresh:
            if(len(sentence) > 0):
                if actions[np.argmax(res)]!=sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if(len(sentence)>5):
            sentence=sentence[-5:]

        cv2.rectangle(image,(0,0),(640,40),(27,114,245),-1)
        cv2.putText(image,' '.join(sentence),(3,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        

        cv2.imshow('Sign Lang',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()