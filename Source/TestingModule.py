import MotionInfluenceGenerator as mig
import MegaBlocksGenerator as cmb
import numpy as np
import cv2

def square(a):
    return (a**2)

def diff(l):
    return (l[0] - l[1])

# Show unusual frame in all actual frame in considered video
def showUnusualActivities(unusual, vid, noOfRows, noOfCols, n):

    unusualFrames = unusual.keys()
    unusualFrames = sorted(unusualFrames)
    print(unusualFrames)
    print(len(unusualFrames))

    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    rows, cols = frame.shape[0], frame.shape[1]
    rowLength = rows/(noOfRows/n)
    colLength = cols/(noOfCols/n)

    # param for scale purpose
    print("Block Size ",(rowLength,colLength))
    count = 0
    screen_res = 980, 520
    scale_width = screen_res[0] / 320
    scale_height = screen_res[1] / 240
    scale = min(scale_width, scale_height)
    window_width = int(320 * scale)
    window_height = int(240 * scale)

    cv2.namedWindow('Unusual Frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unusual Frame',window_width, window_height)
    while 1:
        print(count)
        ret, uFrame = cap.read()
        if (ret == False):
            break
        # if count stands for current frame has been in unusualFrames
        if(count in unusualFrames):
            # get each megablock which identified as one of many block construct an unusual activity at this frame
            # to draw a red rectangle around this block
            for blockNum in unusual[count]:
                print(blockNum)
                x1 = int(blockNum[1] * rowLength)
                y1 = int(blockNum[0] * colLength)
                x2 = int((blockNum[1]+1) * rowLength)
                y2 = int((blockNum[0]+1) * colLength)
                cv2.rectangle(uFrame,(x1,y1),(x2,y2),(0,0,255),1)
            print("Unusual frame number ",str(count))
        cv2.imshow('Unusual Frame',uFrame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        count += 1
    cap.release()
    cv2.destroyAllWindows()


def constructMinDistMatrix(megaBlockMotInfVal,codewords, noOfRows, noOfCols, vid):
    threshold = 5.83682407063e-05
    n = 2
    noMegaBlockRows = int(noOfRows / n)
    noMegaBlockCols = int(noOfCols / n)
    minDistMatrix = np.zeros((len(megaBlockMotInfVal[0][0]), noMegaBlockRows, noMegaBlockCols))

    # get each element in Spatio-temporal motion influence vectors
    for index, val in np.ndenumerate(megaBlockMotInfVal[..., 0]):
        eucledianDist = []

        # get each codewords in 5
        for codeword in codewords[index[0]][index[1]]:

            # we will calculate the difference between each motion influence of current mega block at current frame
            # with each codeword which stand for a intensity state of a motion
            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]), list(codeword)]

            eucDist = (sum(map(square, map(diff, zip(*temp))))) ** 0.5

            eucledianDist.append(eucDist)
        # get the min distance in all value above to set for an element in minDistMatrix
        minDistMatrix[index[2]][index[0]][index[1]] = min(eucledianDist)

    unusual = {}
    for i in range(len(minDistMatrix)):

        # if the max of min distances of all mega block at current frame is still less than threshold
        # it mean this frame can not have an unusual motion
        if (np.amax(minDistMatrix[i]) > threshold):

            unusual[i] = []
            # get each megablock at this frame
            for index, val in np.ndenumerate(minDistMatrix[i]):
                # Adding sequence each mega block at this frame if its motion influence
                if (val > threshold):
                    unusual[i].append((index[0], index[1]))

    print(unusual)
    showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)

# This method is for test purpose when read an actual video
def test_video(vid):

    print ("Test video ", vid)
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)

    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)

    np.save("D:/saved/test/megaBlockMotInfVal_test.npy", megaBlockMotInfVal)

    codewords = np.load("D:/saved/train/codewords_train.npy")

    constructMinDistMatrix(megaBlockMotInfVal, codewords, rows, cols, vid)

# This method is for test purpose when load a megaBlockMotInfVal from
def test_saved_codeword(vid):

    print ("Test video ", vid)
    rows=12
    cols=16

    megaBlockMotInfVal = np.load("D:/saved/test/megaBlockMotInfVal_test.npy")

    codewords = np.load("D:/saved/train/codewords_train.npy")

    constructMinDistMatrix(megaBlockMotInfVal,codewords,rows, cols, vid)

def main():
    testSet = [r"D:/video/test_2.mp4"]
    for video in testSet:
        test_saved_codeword(video)
    print("Testing Done")

if __name__ == '__main__':
    main()

main()