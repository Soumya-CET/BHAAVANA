import cv2
from model import FacialExpressionModel
import numpy as np
############# added for text to speech conversion##############
import pyttsx3
###############################################################
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def my_speak_cloud(my_message):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    engine.say('{}'.format(my_message))
    engine.runAndWait()
    
# returns camera frames along with bounding boxes and predictions
def get_frame(frame):
    #print('get_frame')
    fr = frame
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    print("1")
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)#by default it was  faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    print("2")
    for (x, y, w, h) in faces:
        #print('in for')
        fc = gray_fr[y:y+h, x:x+w]

        roi = cv2.resize(fc, (48, 48))#by default roi = cv2.resize(fc, (48, 48))
        print("3")
        
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        print("5")
        """
        if(pred=="Sad" or pred=="Neutral" or pred=="Happy" or pred=="Fear" or pred=="Disgust" or pred=="Angry" or pred=="Surprise"):
            my_speak_cloud(pred)
        """    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("4")
          


def main():

    video = cv2.VideoCapture('facial_exp.mkv')
    while(video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.destroyAllWindows()
            #video.release()
            break
        
        #print('a')
        get_frame(frame)




if __name__ =='__main__':
    main()

    
