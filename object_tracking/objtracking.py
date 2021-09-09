import cv2

class object_tracking:

    math = __import__("math")
    np = __import__("numpy")
    plt = __import__("matplotlib.pyplot")
    RpyServo = __import__("RpyServo")
    

    # General camera variables

    dpoints_threshold = 0.05 # Minimum distance(in pixels) of movement for it to be recognized for tracking
    pos_threshold = 30 # Minimum distance(in pixels) from object to the center of frame for it to be recognized for tracking
    fov_horizontal = 360 # Horizontal FOV of the camera in degrees of servo rotation (1000 servo degrees = 180 degrees)
    fov_vertical = 260 # Vertical FOV of the camera of servo rotation(1000 servo degrees = 180 degrees)
    size = (640, 480) # Camera resolution


    # Tracking variables

    divider_slow = 12 # Divider for movements of 1 pixel
    divider_fast = 2 # Divider for movements of maximum pixels
    start_position = (500, 500) # Servos starting position, (500,500) is straight forward
    divider_distribution = 'convex' # Default divider distribution, manages compensation for diferent lengths of movements, convex works the best
    rounding = 2 # Rounding digits of coordinates to be passed to servos


    # Technical variables

    max_list_size = 3 # Maximum number of object points the program will hold at a time(SET TO A NUMBER BIGGER THAN 2)


    # NULL variables(DO NOT ALTER)

    points = []
    frames = []
    command = "No command was sent to the servos yet"
    status = 0
    initial_point = None


    # The __init__ function recieves the oject of the detection method, the servo ports(optional, if not defined, the program will only run object detection)
    # and the start position(optional, if not defined, just use the default one, pointing forward)

    def __init__(self, detection_method, servoH_port=None, servoV_port=None, start_position=start_position, servo_pos=("add", "add")):

        self.detection_method = detection_method

        if servoH_port is None or servoV_port is None:
            self.Track = False
            print("Tracking is disabled, only running object detection")
            print("To enable tracking, especify the servo ports in the object_tracking inicialization")
        else:
            self.Track = True
            print("Tracking enabled on the ports:", servoH_port, ",", servoV_port)
            self.servo_pos = servo_pos

            # The servos are objects of Servo_motor class, from the RpyServo library(by yours truly)

            self.servo_horizontal =self.RpyServo.Servo_motor(servoH_port)
            self.servo_vertical = self.RpyServo.Servo_motor(servoV_port)

            self.start_position = start_position

            self.servo_horizontal.set_angle(self.start_position[0])
            self.servo_vertical.set_angle(self.start_position[1])


    # This function simply draws the object position and a rectangle around it

    def draw_datapoints(self, frame):
        # The optical flow tracker needs the points in a numpy array, so it would be easier to just convert it in that code, too bad :)

        new_points = tuple(self.points[-1].ravel())
        old_points = tuple(self.points[-2].ravel())

        overlayed_frame = frame.copy()

        height, width, chennels = overlayed_frame.shape

        overlayed_frame = cv2.line(overlayed_frame, new_points, old_points, (255, 0, 0), 4, -1)

        overlayed_frame = cv2.circle(overlayed_frame, new_points, 7, (170, 0, 170), 5)

        overlayed_frame = cv2.circle(overlayed_frame, (int(width/2),int(height/2)), 7, (255, 0, 0), 5)

        overlayed_frame = cv2.rectangle(overlayed_frame, (int(new_points[0] - self.tracking_area[0]/2), int(new_points[1] - self.tracking_area[1]/2)),
         (int(new_points[0] + self.tracking_area[0]/2), int(new_points[1] + self.tracking_area[1]/2)), (0, 200, 0))

        overlayed_frame = self.draw_text(overlayed_frame, "Tracking")

        return overlayed_frame


    # Pretty simple, just write a text on the frame that shows the current status of tracking

    def draw_text(self, frame, text):
        edited_frame = cv2.putText(frame, text=text, org=(90, 60), fontFace=cv2.QT_FONT_NORMAL, fontScale=2, color=(0,0,255))
        return edited_frame


    # This function does all the communication with the servos, sending them the necessary commands to follow the object, it also aplies the movement divider

    def servo_communication(self, new_points, old_points):
        new_points = new_points.ravel()
        old_points = old_points.ravel()

        width = self.size[0]
        height = self.size[1]

        if abs(new_points[0] - old_points[0]) > self.dpoints_threshold and abs(new_points[1] - old_points[1]) > self.dpoints_threshold and self.status == 1:

            img_center = (int(width / 2), int(height / 2))
            pos_to_center = (img_center[0] - new_points[0], img_center[1] - new_points[1])

            if abs(pos_to_center[0]) > self.pos_threshold or abs(pos_to_center[1]) > self.pos_threshold:

                x_to_center = ((pos_to_center[0] * self.fov_horizontal) / width)
                y_to_center = ((pos_to_center[1] * self.fov_vertical) / height)

                e_x = self.dynamic_divider(self.divider_slow, self.divider_fast, x_to_center, width/2)
                e_y = self.dynamic_divider(self.divider_slow, self.divider_fast, y_to_center, height/2)

                if self.servo_pos[0] == "add":
                    servo_h_angle = self.servo_horizontal.get_angle() + round(x_to_center/e_x, self.rounding)
                else:
                    servo_h_angle = self.servo_horizontal.get_angle() - round(x_to_center/e_x, self.rounding)
                if self.servo_pos[1] == "add":
                    servo_v_angle = self.servo_vertical.get_angle() + round(y_to_center/e_y, self.rounding)
                else:
                    servo_v_angle = self.servo_vertical.get_angle() - round(y_to_center/e_y, self.rounding)

                self.servo_horizontal.set_angle(servo_h_angle)
                self.servo_vertical.set_angle(servo_v_angle)

                self.command = (servo_h_angle, servo_v_angle)


    # This function divides the command to be sent to the servos by a custom number, that is inversely proportional to the size of the movement

    def dynamic_divider(self, ef, ei, deltap, pmax):
        deltap = abs(deltap)
        deltae = ei - ef
        if self.divider_distribution == 'linear':
            e = ((deltae)/pmax)*deltap + ef          
        elif self.divider_distribution == 'concave':         
            e = ((deltae /(pmax**2))*(deltap**2)) - ((deltae /(pmax**2))*deltap) + ef
        elif self.divider_distribution == 'convex':
            e = ((ef - ei)/(pmax**2))*((deltap - pmax)**2) + ei
        else:
            e = ((ef - ei)/(pmax**2))*((deltap - pmax)**2) + ei
        
        return e

    
    # Also pretty self-explanatory, this function receives the initial point(i set it to (300,300) as a default to avoid bugs, probably not a good idea) and adds it to the array of points

    def start(self, frame, initial_point=(300,300)):
        self.initial_point = self.np.array([[initial_point]], dtype=self.np.float32)

        self.frames.append(frame)
        self.points.append(self.initial_point)


    #This function is called every frame, and it handles the whole program, calling all other functions when required

    def update(self, frame):

        # This gets the tracking are from the class of the detection method, it's purely asthetic though, as the only thing that matters is the middle point
        self.tracking_area = self.detection_method.tracking_area

        # This just checks if the number of elements in the points and the frame array is bigger than the allowed size, if so, it simply removes the last one
        if len(self.points) > self.max_list_size:
            self.points.pop(0)

        if len(self.frames) > self.max_list_size:
            self.frames.pop(0)

        self.frames.append(frame)


        # This is the core of the program, it calls the update function in the detection method class to get the new points, the status of the tracking and possibly some error
        new_points, status, error = self.detection_method.run(old_points=self.points[-1], old_frame=self.frames[-2], new_frame=self.frames[-1])

        self.status = status

        # If the tracking is ok, do the whole servo communication(if Track is set), draw the data points and return the overlayed frame
        if status != 0:
            self.points.append(new_points)

            print(self.points[-1])

            if self.Track:
                self.servo_communication(new_points=self.points[-1], old_points=self.points[-2])
            
            self.overlayed_frame = self.draw_datapoints(frame=frame)

            return self.points[-1], self.overlayed_frame
        
        # If not, check if there's an error and add the message accordingly, returning the frame with the text and the last point
        else:
            if error:
                text = "Object lost"   
            else:
                text = "No object set"
            
            self.overlayed_frame = self.draw_text(frame, text)

            return self.points[-1], self.overlayed_frame

