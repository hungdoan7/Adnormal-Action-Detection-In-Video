import cv2
import numpy as np
import os
import opticalFlowOfBlocks as roi

def processVideo():
    os.chdir('/home/hungdoan7/Desktop/Video')
    for file in os.listdir():
        cap = cv2.VideoCapture("./" + file)
        ret, frame1 = cap.read()
        rows, cols = frame1.shape[0], frame1.shape[1]
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # motionInfOfFrames = []
        while 1:
            ret, frame2 = cap.read()
            if (ret == False):
                break
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # motionInfOfFrames = []
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            prvs = next
            opFlowOfBlocks,noOfRowInBlock,noOfColInBlock,blockSize,centreOfBlocks,xBlockSize,yBlockSize = roi.calcOptFlowOfBlocks(mag,ang,next)
            print(centreOfBlocks)
            # motionInfVal = motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize)
            # motionInfOfFrames.append(motionInfVal)
        return xBlockSize,yBlockSize
        #return motionInfOfFrames, xBlockSize,yBlockSize

# a =  [[[1,1],[3,2]],[[5,1],[2,2]]]
# # 2 - 2 - 2
# arr = np.array(a)
# b = arr[...,0]
# print(b)
processVideo()
