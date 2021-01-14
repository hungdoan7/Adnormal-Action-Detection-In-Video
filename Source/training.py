import numpy as np
import motionInfuenceGenerator as mig
import createMegaBlocks as cmb
import app as admin

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def train_from_video(vid):
    '''
        calls all methods to train from the given video
        May return codewords or store them.
    '''
    print ("Training From ", vid)
    MotionInfOfFrames, rows, cols = admin.processVideo(vid)
    print ("Motion Inf Map", len(MotionInfOfFrames))
    #numpy.save("MotionInfluenceMaps", np.array(MotionInfOfFrames), allow_pickle=True, fix_imports=True)
    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save("videos\scene1\megaBlockMotInfVal_set1_p1_train_40-40_k5.npy",megaBlockMotInfVal)
    print(np.amax(megaBlockMotInfVal))
    print(np.amax(reject_outliers(megaBlockMotInfVal)))
    
    codewords = cmb.kmeans(megaBlockMotInfVal)
    np.save("videos\scene1\codewords_set1_p1_train_40-40_k5.npy",codewords)
    print(codewords)
    return
    
if __name__ == '__main__':
    trainingSet = [r"D:/Video/test.avi"]
    for video in trainingSet:
        train_from_video(video)
    print ("Done")
