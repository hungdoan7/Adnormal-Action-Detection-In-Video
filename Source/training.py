import numpy as np
import motionInfuenceGenerator as mig
import createMegaBlocks as cmb

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def train_from_video(vid):
    print ("Training From ", vid)
    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)
    print ("Motion Inf Map", len(MotionInfOfFrames))
    #numpy.save("MotionInfluenceMaps", np.array(MotionInfOfFrames), allow_pickle=True, fix_imports=True)
    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save("D:/saved/test/megaBlockMotInfVal_test.npy", megaBlockMotInfVal)
    print(np.amax(megaBlockMotInfVal))
    print(np.amax(reject_outliers(megaBlockMotInfVal)))
    
    codewords = cmb.kmeans(megaBlockMotInfVal)
    np.save("D:/saved/test/codewords_test.npy",codewords)
    print(codewords)
    return

# def main():
#     trainingSet = [r"D:/Video/test.avi"]
#     for video in trainingSet:
#         train_from_video(video)
#     print("Done")
#
# if __name__ == "__main__":
#     main()

trainingSet = [r"D:/Video/test.mp4"]
for video in trainingSet:
    train_from_video(video)
print("Done")
