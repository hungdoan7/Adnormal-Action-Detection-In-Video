import motionInfuenceGenerator as mig
import megaBlocksGenerator as cmb
import numpy as np
import cv2

def square(a):
    return (a**2)

def diff(l):
    return (l[0] - l[1])

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
        if(count in unusualFrames):
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

    # 6, 8, 101, 8
    for index, val in np.ndenumerate(megaBlockMotInfVal[..., 0]):
        eucledianDist = []

        # each codewords in 5
        for codeword in codewords[index[0]][index[1]]:

            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]), list(codeword)]

            eucDist = (sum(map(square, map(diff, zip(*temp))))) ** 0.5

            eucledianDist.append(eucDist)

        minDistMatrix[index[2]][index[0]][index[1]] = min(eucledianDist)

    unusual = {}
    for i in range(len(minDistMatrix)):
        if (np.amax(minDistMatrix[i]) > threshold):

            unusual[i] = []
            for index, val in np.ndenumerate(minDistMatrix[i]):

                if (val > threshold):
                    unusual[i].append((index[0], index[1]))

    print(unusual)
    showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)
    
def test_video(vid):

    print ("Test video ", vid)
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)

    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)

    np.save("D:/saved/test/megaBlockMotInfVal_test.npy", megaBlockMotInfVal)

    codewords = np.load("D:/saved/train/codewords_train.npy")

    constructMinDistMatrix(megaBlockMotInfVal, codewords, rows, cols, vid)

def test_saved_codeword(vid):

    print ("Test video ", vid)
    rows=12
    cols=16

    megaBlockMotInfVal = np.load("D:/saved/test/megaBlockMotInfVal_test.npy")

    codewords = np.load("D:/saved/train/codewords_train.npy")

    constructMinDistMatrix(megaBlockMotInfVal,codewords,rows, cols, vid)

def main():
    testSet = [r"D:/video/test_1.avi"]
    for video in testSet:
        test_saved_codeword(video)
    print("Testing Done")

if __name__ == '__main__':
    main()

main()