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
    #k-means
    cluster_n = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    codewords = np.zeros((len(megaBlockMotInfVal),len(megaBlockMotInfVal[0]),cluster_n,8))

    #codewords = []
    #print("Mega blocks ",megaBlockMotInfVal)
    for row in range(len(megaBlockMotInfVal)):
        for col in range(len(megaBlockMotInfVal[row])):
            #print("megaBlockMotInfVal ",(row,col),"/n/n",megaBlockMotInfVal[row][col])
            
            ret, labels, cw = cv2.kmeans(np.float32(megaBlockMotInfVal[row][col]), cluster_n, None, criteria,10,flags)
            #print(ret)
            #if(ret == False):
            #    print("K-means failed. Please try again")
            codewords[row][col] = cw
            
    return(codewords)
