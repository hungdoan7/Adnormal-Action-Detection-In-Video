import numpy as np
import math

def calcOptFlowOfBlocks(mag, angle, grayImg):

    # rows represent for the number rows of an image
    # rows represent for the number cols of an image
    rows = grayImg.shape[0]
    cols = grayImg.shape[1]

    # We declare that every block will have a fixed size 20x20
    noOfRowInBlock = 20
    noOfColInBlock = 20

    # We calculate the actual row number and col number of blocks if we organize it as a new array which was constructed by 20x20 = 400 pixels
    xBlockSize = int (rows / noOfRowInBlock)
    yBlockSize = int (cols / noOfColInBlock)

    # Initial a 3 dimension array which actual is a 2d array has the size fit the blocks size and each its element is a smaller 1 dimension array
    # holding 2 element use to hold magnitude of each pixel and θ represents the direction in which the each pixel has moved relative to the
    # corresponding block in the previous frame
    opFlowOfBlocks = np.zeros((xBlockSize,yBlockSize,2))

    # Select each index and value from 2d array holding magnitude of all the pixel in a frame
    for index,value in np.ndenumerate(mag):
        # Identify that the current pixel is related to what block
        # through out calculating the index of corresponding block in opFlowOfBlocks array
        xBlockIndex = int (index[0]/noOfRowInBlock)
        yBlockIndex = int (index[1]/noOfColInBlock)

        # Sum the magnitude and angle of direction in optical value in current pixel to its block it belong
        opFlowOfBlocks[xBlockIndex][yBlockIndex][0] += mag[index[0]][index[1]]
        opFlowOfBlocks[xBlockIndex][yBlockIndex][1] += angle[index[0]][index[1]]

    # Initial a 3 dimension array has shape equal to opFlowOfBlocks
    centreOfBlocks = np.zeros((xBlockSize,yBlockSize,2))
    # Select each index and value from 2d array holding magnitude of all the pixel in a block
    for index,value in np.ndenumerate(opFlowOfBlocks):

        # Calculate the average value of current optical flow value element (magnitude or direction) of current block
        opFlowOfBlocks[index[0]][index[1]][index[2]] = float(value)/(noOfRowInBlock*noOfColInBlock)
        val = opFlowOfBlocks[index[0]][index[1]][index[2]]

        # Angle attribute
        # Using this val above variable to calculate the index which lead to the direction of one of 8 blocks around each block
        if(index[2] == 1):
            angInDeg = math.degrees(val)
            if(angInDeg > 337.5):
                k = 0
            else:
                q = angInDeg//22.5
                a1 = q*22.5
                q1 = angInDeg - a1
                a2 = (q+2)*22.5
                q2 =  a2 - angInDeg
                if(q1 < q2):
                    k = int(round(a1/45))
                else:
                    k = int(round(a2/45))
            opFlowOfBlocks[index[0]][index[1]][index[2]] = k
            print(opFlowOfBlocks[index[0]][index[1]][index[2]])
        else:
            # Calculate the centre pixel of each block
            # This index of the centre of each block will be used to calculate Euclidean Dist between it and another block
            # x, y is the index of the centre pixel in the 2D array which structure a frame
            x = ((index[0] + 1)*noOfRowInBlock)-(noOfRowInBlock/2)
            y = ((index[1] + 1)*noOfColInBlock)-(noOfColInBlock/2)
            centreOfBlocks[index[0]][index[1]][0] = x
            centreOfBlocks[index[0]][index[1]][1] = y
    return opFlowOfBlocks,noOfRowInBlock,noOfColInBlock,noOfRowInBlock*noOfColInBlock,centreOfBlocks,xBlockSize,yBlockSize

