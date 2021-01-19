import numpy as np
import MotionInfluenceGenerator as mig
import MegaBlocksGenerator as cmb

def train_from_video(vid):
    print ("Training From ", vid)
    # Get motion influence map
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)

    # Get the Spatio-temporal motion influence vector of all megablocks
    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save("D:/saved/train/megaBlockMotInfVal_train.npy", megaBlockMotInfVal)

    # Get centers (codewords) of k clusters when we consider the clustering problems
    # In which we want to identify k stage of motion like 5 cluster of all motion influence of a megablock in all frame
    # So we can catch the almost correct motion intensity of this perspective
    codewords = cmb.kmeans(megaBlockMotInfVal)

    # Save centers aka codewords into a binary file to use later
    np.save("D:/saved/train/codewords_train.npy",codewords)
    print(codewords)
    return

def main():

    # Just consider this module as a starting point of this time and below is the function calling
    # For training method we only use many usual video in a perspective aspect
    trainingSet = ["D:/Video/train_1.avi"]
    for video in trainingSet:
        train_from_video(video)
    print("Trainning Done")

if __name__ == "__main__":
    main()


