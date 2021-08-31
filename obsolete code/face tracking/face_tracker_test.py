import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def face_detection(frame, area_thresh, middle_point):
    final_rects = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = faceCascade.detectMultiScale(gray, scaleFactor=1.05,
        minNeighbors=9, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    
    for rect in rects:
        if rect[2]*rect[3] > area_thresh:
            final_rects.append(rect)
    
    if len(final_rects) > 0:
        face_detected = True 
        rect = rects[0]
        tracking_area = ()
        middle_point = (rect[0] + (rect[2]/2) , rect[1] + (rect[3]/2))

        return face_detected, middle_point, rect

    else: 
        return False, middle_point, (0,0)


def tracking(frame, middle_point, tracker, tracker_status):
    if tracker_status== 0:
        tracker.init()
        tracker_status = 1
        return tracker_status, (0,0)

    tracker_status, rect = tracker.update(frame)

    middle_point = (rect[0] + (rect[2]/2) , rect[1] + (rect[3]/2))

    return tracker_status, middle_point, rect


# Technical variables
frame_num = 0
face_detected = False
middle_point = (0,0)

# General variables
frames_to_detec = 60
cnt_area_thresh = 300

tracker = cv2.TrackerKCF_create()
tracker_status = 0
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()

    if frame_num == frames_to_detec or face_detected == False:
        face_detected, middle_point, rect = face_detection(frame, 300, middle_point)
    
    else:
        face_detected, middle_point, rect = tracking(frame, middle_point, tracker, face_detected)

    points_rect = ( (rect[0], rect[1]) , (rect[0] + rect[2], rect[1] + rect[3]))

    ol_frame = cv2.rectangle(frame, points_rect[0], points_rect[1], (0,255,0), 5)

    ol_frame = cv2.circle(ol_frame, middle_point, 5, (255,0,0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
            print("Stop key pressed, quitting program...")
            exit()



