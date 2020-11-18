import cv2
import os
#Point to video address
#Ponit a varibale to separated video folder in Desktop
os.chdir('/home/hungdoan7/Desktop/Video')

for file in os.listdir():
    # Split the first part of the name before "." to using for build a folder name below
    array_name = file.split(".")
    video_name = array_name[0]

    # Read the video from specified path
    cam = cv2.VideoCapture("./" + file)

    try:
        # creating a folder named data for each video (int current folder) we will extract frame from them
        if not os.path.exists('./data/'+ video_name):
            os.makedirs('./data/' + video_name)
            # Counter for id of each frame
            currentframe = 0

            while(True):
                # reading from frame
                # ret is a boolean variable to detect as if any frame after this current frame everytime we read a frame
                ret, frame = cam.read()
                if ret:
                    # if video is still left continue creating images
                    #declare a string var to store the path which will locate the current frame
                    name = './data/'+ video_name +'/frame' + str(currentframe) + '.jpg'
                    print ('Creating...' + name)

                    # writing the extracted images to that above path
                    cv2.imwrite(name, frame)

                    # increasing counter
                    currentframe += 1
                else:
                    break
            # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()
        else:
            # This video has been already distracted
            print("File " + video_name + ".mp4 is extracted")
    # if not created then raise error
    except OSError:
        print ('Error to Creating directory of data')
