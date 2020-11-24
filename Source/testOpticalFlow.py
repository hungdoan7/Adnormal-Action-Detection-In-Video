import math
import cv2
import os
import numpy as np
import Source.opticalFlowOfBlocks as roi

def getThresholdDistance(mag,blockSize):
    return mag*blockSize

def getThresholdAngle(ang):
    tAngle = float(math.pi)/2
    return ang+tAngle, ang-tAngle

def getCentreOfBlock(blck1Indx,blck2Indx,centreOfBlocks):
    x1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][0]
    y1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][1]
    x2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][0]
    y2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][1]
    if (x1 != x2):
        slope = float((y2-y1)/(x2-x1))
    else:
        slope = float("inf")
    return (x1,y1),(x2,y2),slope

def calcEuclideanDist(x1,y1,x2,y2):
    dist = float(((x2-x1)**2 + (y2-y1)**2)**0.5)
    return dist

def angleBtw2Blocks(ang1,ang2):
    if(ang1-ang2 < 0):
        ang1InDeg = math.degrees(ang1)
        ang2InDeg = math.degrees(ang2)
        return math.radians(360 - (ang1InDeg-ang2InDeg))
    return ang1 - ang2

def motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize):
    # global frameNo
    motionInfVal = np.zeros((xBlockSize,yBlockSize,8))

    for index,value in np.ndenumerate(opFlowOfBlocks[...,0]):
        Td = getThresholdDistance(opFlowOfBlocks[index[0]][index[1]][0],blockSize)
        k = opFlowOfBlocks[index[0]][index[1]][1]
        posFi, negFi = getThresholdAngle(math.radians(45*(k)))

        for ind,val in np.ndenumerate(opFlowOfBlocks[...,0]):
            if(index != ind):
                (x1,y1),(x2,y2), slope = getCentreOfBlock(index,ind,centreOfBlocks)
                euclideanDist = calcEuclideanDist(x1,y1,x2,y2)

                if(euclideanDist < Td):
                    angWithXAxis = math.atan(slope)
                    angBtwTwoBlocks = angleBtw2Blocks(math.radians(45*(k)),angWithXAxis)

                    if(negFi < angBtwTwoBlocks and angBtwTwoBlocks < posFi):
                        motionInfVal[ind[0]][ind[1]][int(opFlowOfBlocks[index[0]][index[1]][1])] += math.exp(-1*(float(euclideanDist)/opFlowOfBlocks[index[0]][index[1]][0]))
    #print("Frame number ", frameNo)
    # frameNo += 1
    return motionInfVal


def processVideo():
    # Set the directory which have videos we need
    os.chdir('/home/hungdoan7/Desktop/Video')
    for file in os.listdir():
        cap = cv2.VideoCapture("./" + file)
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        ret, frame1 = cap.read()
        if (ret == False):
            break
        # Change image to gray image in which each pixel has a value between 0 and 255
        # And also set the prvs variable to hold the previous image
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        motionInfOfFrames = []
        while 1:
            ret, frame2 = cap.read()
            if (ret == False):
                break
            # Change image to gray image in which each pixel has a value between 0 and 255
            # And also set the next variable to hold the next image
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Calculating dense optical flow by Farneback method
            # The output is a array have 3 dimension, the 2 first dimensions organize as the structure of pixels in an image
            # The third dimension holding 2 value stands for magnitude and direction when compute optical flow of a pixel compare to itself in previous frame
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Computes the magnitude and angle of the 2D vectors
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            # Calculate the optical flow of each block in an image
            opFlowOfBlocks,noOfRowInBlock,noOfColInBlock,blockSize,centreOfBlocks,xBlockSize,yBlockSize = roi.calcOptFlowOfBlocks(mag,ang,next)
            motionInfVal = motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize)
            motionInfOfFrames.append(motionInfVal)
            print(motionInfVal.shape)
            # Update or in other words is setting previous image equal current image
            prvs = next
        return motionInfOfFrames,xBlockSize,yBlockSize
#  Just consider this module as a starting point of this time and below is the function calling
motionInfVal,xBlockSize,yBlockSize = processVideo()
print(motionInfVal.shape)
