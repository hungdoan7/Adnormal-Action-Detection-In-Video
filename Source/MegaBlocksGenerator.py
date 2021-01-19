import cv2

import numpy as np

def createMegaBlocks(motionInfoOfFrames,noOfRows,noOfCols):

    # so to create the mega block which has size of 2x2 cube (each element is a block), we init some param for the size purpose
    n = 2
    noOfMegaBlockRow = int(noOfRows/n)
    noOfMegaBlockCol = int(noOfCols/n)

    # we create a 4 dimensions array in which one position mega block (identify by the 2 first dimensions) hold 1 array (2th dimension) of t element
    # standing for t frame in the considered video and in this we have a moition influence vector (4th dimension) of this mega block at this frame.
    megaBlockMotInfVal = np.zeros((noOfMegaBlockRow,noOfMegaBlockCol,len(motionInfoOfFrames),8))
    frameCounter = 0

    # Motion influence vector in each frame of each megablock
    for frame in motionInfoOfFrames:

        # Each
        for index,val in np.ndenumerate(frame[...,0]):

            indexOfMegaBlockRow = int(index[0]/n)
            indexOfMegaBlockCol = int(index[1]/n)

            temp = [list(megaBlockMotInfVal[indexOfMegaBlockRow][indexOfMegaBlockCol][frameCounter]), list(frame[index[0]][index[1]])]

            # Sum each element (motion influence of each block inside a megablock) to construct motion influence of each megablock in this frame
            megaBlockMotInfVal[indexOfMegaBlockRow][indexOfMegaBlockCol][frameCounter] = np.array(list(map(sum, zip(*temp))))

        frameCounter += 1

    return megaBlockMotInfVal

def kmeans(megaBlockMotInfVal):

    # we can clustering more cluster but the difference accurracy between them is not so much
    # We define 5 cluster stand for 5 motion stage of a motion in a megablock in all frames so we can catch the right intensities.
    cluster_n = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    noOfMegaBlockRows = len(megaBlockMotInfVal)
    noOfMegaBlockCols = len(megaBlockMotInfVal[0])

    # The 4 dimensions array the 2 first dimension is about the location of current mega block, each element inside is a center of a cluster we want to indentify.
    # in wich is hold the average motion influence of all frame of this mega block in this current cluster.
    codewords = np.zeros((noOfMegaBlockRows, noOfMegaBlockCols, cluster_n, 8))

    print("print out len of array", len(megaBlockMotInfVal))
    print("print out len of array[0]", len(megaBlockMotInfVal[0]))

    for row in range(noOfMegaBlockRows):
        for col in range(noOfMegaBlockCols):
            
            compactness, labels, cw = cv2.kmeans(np.float32(megaBlockMotInfVal[row][col]), cluster_n, None, criteria, 10, flags)
            codewords[row][col] = cw
            
    return(codewords)
