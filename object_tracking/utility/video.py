import cv2

class video_capture:
    cv2 = __import__("cv2")

    frame = None
    stop_key = 'q'
    window_name = " Main"

    def __init__(self, camera_index, window_name=window_name, flipped=False):
        self.camera_index = camera_index
        self.cap = self.cv2.VideoCapture(camera_index)
        self.window_name = window_name
        self.flipped = flipped

        if not self.cap.isOpened():
            print("Cannot open camera, check if the camera_index is correct andtry again")
            exit()


    def get(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Cannot capture frame from camera " + str(self.camera_index) + " (maybe the camera was disconnected or the stream ended)")
            exit()
        
        if self.flipped:
            frame = self.cv2.flip(frame, 0)

        self.frame = frame

        return frame

    
    def release(self):
        self.cap.release()
        self.cv2.destroyAllwindows()
        exit()

    
    def display(self, frame=frame, window_name = window_name, stop_key = stop_key):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord(stop_key):
            print("Stop key pressed, quitting program...")
            exit()

'''
class detect_events:
    cv2 = __import__('cv2')

    window_name = 'Main'
    event = cv2.EVENT_LBUTTONDOWN
    point_was_selected = False
    points = ()


    def __init__(self, window_name=window_name, event=event):
        self.cv2.namedWindow(window_name)
        self.cv2.setMouseCallback(window_name, self.handle_point_selection)
        self.event = event
        self.window_name = window_name


    def handle_point_selection(self, event, x, y, flags, params):
        if event == self.event:
            self.point_was_selected = True
            self.points = (x, y)


    def update(self):
        if self.point_was_selected:
            print("event detected")
            return True, self.points
        else:
            return False, self.points
'''


def display_image(frame, window_name="Main", stop_key='q'):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord(stop_key):
            print("Stop key pressed, quitting program...")
            exit()
