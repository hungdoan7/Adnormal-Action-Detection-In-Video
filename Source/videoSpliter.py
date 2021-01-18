import cv2

# we wanna make a func that help us achieve getting a correct resolution video (in each frame) we need

def spl():
    cap = cv2.VideoCapture(r'Crowd-Activity-All.avi')

    # start and end index frame to split
    start, end = 7658,7738

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # resolution we want is 320x240
    out = cv2.VideoWriter('3_test3.avi',fourcc, 30.0, (320, 240))
    cap.set(cv2.CAP_PROP_POS_FRAMES , start)
    while end-start>0:
        print("loop")
        ret, frame = cap.read()
        frame = cv2.resize(frame,(320, 240))
        out.write(frame)
        start+=1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

spl()
