import cv2 
import os 

#Point to video address
default_address = os.getcwd();
os.chdir('D:\\Truc\\DA3\\Video')
for file in os.listdir():
    array_name = file.split(".")
    video_name = array_name[0]
    # Read the video from specified path 
    cam = cv2.VideoCapture(".\\" + file)
  
    try: 
        
        # creating a folder named data 
        if not os.path.exists(default_address + '\\data\\'+ video_name): 
            os.makedirs(default_address + '\\data\\' + video_name)
            # frame 
            currentframe = 0
  
            while(True): 
      
                # reading from frame 
                ret,frame = cam.read() 
  
                if ret: 
                    # if video is still left continue creating images 
                    name = default_address + '\\data\\'+ video_name +'\\frame' + str(currentframe) + '.jpg'
                    print ('Creating...' + name) 
  
                    # writing the extracted images 
                    cv2.imwrite(name, frame) 
  
                    # increasing counter so that it will 
                    # show how many frames are created 
                    currentframe += 1
                else: 
                    break
  
            # Release all space and windows once done 
            cam.release() 
            cv2.destroyAllWindows() 
        else:
            print("File " + video_name + ".mp4 is extracted")
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
  