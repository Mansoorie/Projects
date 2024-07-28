import cv2
import numpy as np
import time

class Detector:
    def __init__(self, videopath, configpath, modelpath, classespath):
        self.videopath = videopath
        self.configpath = configpath
        self.modelpath = modelpath
        self.classespath = classespath

        ##########################
        self.net = cv2.dnn.DetectionModel(self.modelpath, self.configpath)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.setInputSize(320, 320)  # Corrected line
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        self.readclasses()

    def setInputSize(self, width, height):  # Added setInputSize method
        self.net.setInputSize(width, height)

    def readclasses(self):
        with open(self.classespath, 'r') as f:
            self.classeslist = f.read().splitlines()

        self.classeslist.insert(0, 'BACKGROUND')

        print(self.classeslist)

    def onvideo(self):
        cap = cv2.VideoCapture(self.videopath)


        if(cap.isOpened()==False):
            print("EROOR OPENING FILE ....")
            return

        (success,image)= cap.read()

        while success:
           classLabelIDs, confidences, bboxs = self.net.detect(image , confThreshold =0.5)
           confidences = list(np.array(confidences).reshape(1,-1)[0])
           confidences = list(map(float,confidences))

           bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold= 0.2)

           if len(bboxIdx) != 0:
               for i in range(0,len(bboxIdx)):

                   bbox = bboxs[np.squeeze(bboxIdx[i])]
                   classConfidence = confidences[np.squeeze(bboxIdx[i])]
                   classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                   classlabel = self.classeslist[classLabelID]

                   displaytext = "{}:{:.4f}".format(classlabel, classConfidence)


                   x,y,w,h = bbox

                   cv2.rectangle(image, (x,y), (x+w,y+h), color=(0,0,255), thickness= 1)
                   cv2.putText(image, displaytext, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, color=(0,0,255), thickness=2)

           cv2.imshow("RESULT", image)

           key= cv2.waitKey(1) & 0xFF
           if key == ord("q"):
               break
           (success,image) = cap.read()
        cv2.destroyAllWindows()








