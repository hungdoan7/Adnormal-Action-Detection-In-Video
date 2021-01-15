import cv2
import os
import opticalFlowOfBlocks as roi
import motionInfuenceGenerator

def processVideo():

    # Set the directory which have videos we need
    os.chdir(r'D:/Video')

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
            motionInfVal = motionInfuenceGenerator.motionInMapGenerator(opFlowOfBlocks,blockSize,centreOfBlocks,xBlockSize,yBlockSize)
            motionInfOfFrames.append(motionInfVal)

            # Update or in other words is setting previous image equal current image
            prvs = next

        return motionInfOfFrames, xBlockSize, yBlockSize


