import math
import numpy as np
import cv2
import opticalFlowOfBlocks as roi

def getThresholdDistance(mag, blockSize):
    return mag*blockSize

def getThresholdAngle(ang):
    tAngle = float(math.pi)/2
    return ang+tAngle, ang-tAngle

def getCentreOfBlock(blck1Indx,blck2Indx,centreOfBlocks):

    # get sequentially the x and y position of centre of 2 blocks to calculate the slope
    x1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][0]
    y1 = centreOfBlocks[blck1Indx[0]][blck1Indx[1]][1]
    x2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][0]
    y2 = centreOfBlocks[blck2Indx[0]][blck2Indx[1]][1]

    # if the first block and second block has the different row position
    # we will calculate the slope
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
    # create a three dimension array with the third dimension holds 8 value related to its influence to 8 relatived direction around
    motionInfVal = np.zeros((xBlockSize,yBlockSize,8))

    # select each element in the whole blocks array
    for index,value in np.ndenumerate(opFlowOfBlocks[...,0]):

        # Calculate the sum of the whole magnitude of optical flow of this block by multiply with total number pixel in each block
        # as a distance threshold
        Td = getThresholdDistance(opFlowOfBlocks[index[0]][index[1]][0],blockSize)

        # Get the angle of optical flow in this block
        k = opFlowOfBlocks[index[0]][index[1]][1]

        # Get the current angle + pi/2 and current angle - pi/2
        # as angle threshold
        posFi, negFi = getThresholdAngle(math.radians(45*(k)))

        # We will calculate the influence of this block to the rest blocks
        # First block is the
        for ind, val in np.ndenumerate(opFlowOfBlocks[...,0]):
            if(index != ind):

                # Calculate the relatived position of 2 blocks and slope between 2 blocks
                (x1,y1),(x2,y2), slope = getCentreOfBlock(index,ind,centreOfBlocks)
                # Calculate the relatived distance of 2 blocks and slope between 2 blocks
                euclideanDist = calcEuclideanDist(x1,y1,x2,y2)

                # if the relatived distance of 2 blocks and slope between 2 blocks < the distance threshold
                if(euclideanDist < Td):
                    # Calculate the angle of this slope (between 2 blocks) with the x axis
                    angWithXAxis = math.atan(slope)

                    # Calculate the relatived angle between 2 blocks
                    angBtwTwoBlocks = angleBtw2Blocks(math.radians(45*(k)),angWithXAxis)

                    # If the  recent angle between 2 blocks is satisfy,
                    # we will calculate the influence of the first block to the second block
                    # and store it in the right direction as well as the corresponding
                    if(negFi < angBtwTwoBlocks and angBtwTwoBlocks < posFi):
                        motionInfVal[ind[0]][ind[1]][int(opFlowOfBlocks[index[0]][index[1]][1])] += math.exp(-1*(float(euclideanDist)/opFlowOfBlocks[index[0]][index[1]][0]))
    return motionInfVal


def getMotionInfuenceMap(vid):

    global frameNo

    frameNo = 0
    cap = cv2.VideoCapture(vid)
    ret, frame1 = cap.read()
    rows, cols = frame1.shape[0], frame1.shape[1]
    print(rows, cols)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    motionInfOfFrames = []
    count = 0
    while 1:
        ret, frame2 = cap.read()
        if (ret == False):
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        prvs = next
        opFlowOfBlocks, noOfRowInBlock, noOfColInBlock, blockSize, centreOfBlocks, xBlockSize, yBlockSize = roi.calcOptFlowOfBlocks(
            mag, ang, next)
        motionInfVal = motionInMapGenerator(opFlowOfBlocks, blockSize, centreOfBlocks, xBlockSize, yBlockSize)
        motionInfOfFrames.append(motionInfVal)

        # if(count == 622):
        #    break
        count += 1
        print(count)
    return motionInfOfFrames, xBlockSize, yBlockSize