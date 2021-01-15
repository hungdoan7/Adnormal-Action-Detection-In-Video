import cv2

import numpy as np

def createMegaBlocks(motionInfoOfFrames,noOfRows,noOfCols):

    n = 2
    noOfMegaBlockRow = int(noOfRows/n)
    noOfMegaBlockCol = int(noOfCols/n)

    megaBlockMotInfVal = np.zeros((noOfMegaBlockRow,noOfMegaBlockCol,len(motionInfoOfFrames),8))
    frameCounter = 0
    
    for frame in motionInfoOfFrames:
        
        for index,val in np.ndenumerate(frame[...,0]):

            indexOfMegaBlockRow = int(index[0]/n)
            indexOfMegaBlockCol = int(index[1]/n)

            temp = [list(megaBlockMotInfVal[indexOfMegaBlockRow][indexOfMegaBlockCol][frameCounter]), list(frame[index[0]][index[1]])]

            megaBlockMotInfVal[indexOfMegaBlockRow][indexOfMegaBlockCol][frameCounter] = np.array(list(map(sum, zip(*temp))))

        frameCounter += 1

    return megaBlockMotInfVal

def kmeans(megaBlockMotInfVal):

    cluster_n = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    noOfMegaBlockRows = len(megaBlockMotInfVal)
    noOfMegaBlockCols = len(megaBlockMotInfVal[0])

    codewords = np.zeros(noOfMegaBlockRows, noOfMegaBlockCols, cluster_n, 8)

    print("print out len of array", len(megaBlockMotInfVal))
    print("print out len of array[0]", len(megaBlockMotInfVal[0]))

    for row in range(noOfMegaBlockRows):
        for col in range(noOfMegaBlockCols):
            
            compactness, labels, cw = cv2.kmeans(np.float32(megaBlockMotInfVal[row][col]), cluster_n, None, criteria, 10, flags)
            codewords[row][col] = cw
            
    return(codewords)
