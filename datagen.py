import cv2
import os
import time


def func(t):
    start_time = time.time()
    seconds = t

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > seconds:
            print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
            break

IMG_PATH='data'
name="Shyam"
round="3"

labels=[]
for i in range(97,123):
    labels.append(chr(i))

cap=cv2.VideoCapture(0)
print('Get ready..')
time.sleep(3)
for label in labels:
    if not os.path.exists(os.path.join(IMG_PATH,label)):
        os.makedirs(os.path.join(IMG_PATH,label))
    
    # cap=cv2.VideoCapture(0)
    print('Collecting image for: ',label)
    time.sleep(5)
    
    
    # start_time = time.time()
    # seconds = 4

    # while True:
    #     current_time = time.time()
    #     elapsed_time = current_time - start_time
    #     ret,frame=cap.read()
    #     cv2.imshow('frame',frame)
    #     if elapsed_time > seconds:
    #         print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
    #         break
    ret,frame=cap.read()
    cv2.imshow('frame',frame)

    print('writing...')
    cv2.imwrite(os.path.join(IMG_PATH,label,label+'_'+name+'_'+round+'.png'),frame)

    if(cv2.waitKey(1) & 0xFF==ord("q")):
        break
    # cap.release()
cap.release()