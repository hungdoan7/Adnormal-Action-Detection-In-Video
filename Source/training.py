import numpy as np
import motionInfuenceGenerator as mig
import megaBlocksGenerator as cmb

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def train_from_video(vid):
    print ("Training From ", vid)
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)

    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save("D:/saved/train/megaBlockMotInfVal_train.npy", megaBlockMotInfVal)
    print(np.amax(megaBlockMotInfVal))
    print(np.amax(reject_outliers(megaBlockMotInfVal)))
    
    codewords = cmb.kmeans(megaBlockMotInfVal)
    np.save("D:/saved/train/codewords_train.npy",codewords)
    print(codewords)
    return

def main():
    # Just consider this module as a starting point of this time and below is the function calling
    trainingSet = ["D:/Video/train_1.avi"]
    for video in trainingSet:
        train_from_video(video)
    print("Trainning Done")

if __name__ == "__main__":
    main()


