import cv2
import imutils
import numpy as np
from PyQt5 import QtGui, QtWidgets
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from load_and_process import preprocess_input


class Emotion_Rec:
    def __init__(self, model_path=None):

        # load the data and image parameters
        detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

        if model_path == None:  # use default model if the path is none
            emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
        else:
            emotion_model_path = model_path

        # load face classifier
        self.face_detection = cv2.CascadeClassifier(detection_model_path)  # cascade classifier

        # load the emotion model
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        # emotion catalog
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]

    def run(self, frame_in, canvas, label_face, label_result):
        # frame_in camera frame
        # canvas background image
        # label_face 
        # label_result 

        # adjust the frame size
        frame = imutils.resize(frame_in, width=300)  # resize the frame
        # frame = cv2.resize(frame, (300,300))  # resize the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change to the gray frame

        # detect the face
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        preds = []  # predict result
        label = None  # prediction label
        (fX, fY, fW, fH) = None, None, None, None  # face location
        frameClone = frame.copy()  # copy the frame

        if len(faces) > 0:
            # sorted the human face by ROI value.
            faces = sorted(faces, reverse=False, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # sort by area

            for i in range(len(faces)):  
                
                (fX, fY, fW, fH) = faces[i]

                # extract the interesting ROI area, and load into CNN.
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])
                roi = preprocess_input(roi)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # model predict the probability of each catalog
                preds = self.emotion_classifier.predict(roi)[0]
                # emotion_probability = np.max(preds)  # maximum probability
                label = self.EMOTIONS[preds.argmax()]  # choose the emotion catalog with maximum probability
                # circle the face area
                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)

        # canvas = 255* np.ones((250, 300, 3), dtype="uint8")
        # canvas = cv2.imread('slice.png', flags=cv2.IMREAD_UNCHANGED)

        for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
            # show the probability of each catalog
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # plot the probability graph
            w = int(prob * 300) + 7
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (224, 200, 130), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # adjust the image size
        frameClone = cv2.resize(frameClone, (420, 280))

        # show the face in QT window
        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()

        # show the result with labels.
        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_result.setPixmap(QtGui.QPixmap.fromImage(showImage))

        return (label)
