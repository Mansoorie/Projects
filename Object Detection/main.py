from Detector import *
import os

def main():
    videopath = "test3.mp4"
    configpath = os.path.join("ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")

    modelpath = os.path.join("frozen_inference_graph.pb")
    classespath = os.path.join("coco.names")

    detector = Detector(videopath,configpath,modelpath,classespath)
    detector.onvideo()
if __name__ == '__main__':
    main()

