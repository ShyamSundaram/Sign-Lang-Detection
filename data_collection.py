import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_landmarks(image,results):
    #mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

def extract_keypoints(results):
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])

def collect_data():
    DATA_PATH=os.path.join('MP_DATA2')
    actions=[]
    for i in range(97,123):
        actions.append(chr(i))
    no_sequences=10
    sequence_length=10

    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
            except:
                pass

    cap=cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        for action in actions[0:4]:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):

                    ret,frame=cap.read()

                    image,results=mediapipe_detection(frame,holistic)
                    #print(results)

                    draw_landmarks(image,results)

                    if(frame_num==0):
                        cv2.putText(image,'STARTING COLLECTION FOR {} VIDEO NO {}'.format(action,sequence),(120,200), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                        cv2.putText(image,'Collecting frames for {}'.format(action),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed',image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image,'Collecting frames for {} Video no {}'.format(action,sequence),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed',image)


                    keypoints=extract_keypoints(results)
                    np.save(os.path.join(DATA_PATH,action,str(sequence),str(frame_num)),keypoints)

                    cv2.imshow('OpenCV Feed',image)
                    
                    if(cv2.waitKey(10) & 0xFF==ord('q')):
                        break
                
        cap.release()
        cv2.destroyAllWindows()