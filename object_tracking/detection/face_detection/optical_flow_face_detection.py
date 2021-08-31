
class face_detection:

        cv2 = __import__("cv2")
        np = __import__("numpy")
        
        tracking_area = (100,100) # Quantidade de pixeis rastreados ao redor do pixel central
        bounding_area = (0,0,0,0)
        max_level = 2 # Nível de pirâmide usado
        epsilon = 0.05 # Valores menores de epsilon deixam o programa mais rápido, porém menos preciso
        iterations = 5 # Número de iterações do optical flow, quanto mais iteracoes, mais preciso é a detecção, porém roda mais devagar
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        max_frame_face_track = 3000


        #Technical variables(avoid changing)
        points = []
        frame_count = 0
        face_detected = False
 
        
        def __init__(self, tracking_area = tracking_area, max_level = max_level, epsilon = epsilon, iterations = iterations):
            self.tracking_area = tracking_area
            self.max_level = max_level
            self.epsilon = epsilon
            self.iterations = iterations
        

        def gray_scale(self, frame):
            frame_gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)

            return frame_gray
        

        def optical_flow_tracking(self, old_frame, new_frame, old_points):
            lk_params = dict( winSize =(self.tracking_area[0], self.tracking_area[1]),
                maxLevel=self.max_level, criteria=( self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT, self.iterations, self.epsilon))
            new_gray = self.gray_scale(new_frame)
            old_gray = self.gray_scale(old_frame)
            new_points, status, error = self.cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)
            status = status.ravel()[0]
            if status == 0:
                new_points = (0,0)
                new_points = self.np.array([[new_points[0], new_points[1]]], dtype=self.np.float32)
            return new_points, status, error
        

        def face_detection(self, frame):
            print("Face detection running")
            gray = self.gray_scale(frame)

            rects = self.faceCascade.detectMultiScale(gray, scaleFactor=1.05,
                minNeighbors=9, minSize=(30, 30),
                flags=self.cv2.CASCADE_SCALE_IMAGE)

            if len(rects) > 0:
                self.face_detected = True

                rect = rects[0]

                self.bounding_area = rect

                rect = (rect[0], rect[1], rect[0] + rect[2], rect[1]+rect[3])

                self.tracking_area = ( abs(rect[2] - rect[0]), abs(rect[3] - rect[1]) ) 

                middle_point = ((rect[0] + rect[2])/2 , (rect[1] + rect[3])/2 )
                
                status = 1
                error = 0       
            else:
                middle_point = (0,0)
                status = 0
                error = 0
            new_points = self.np.array([[middle_point[0], middle_point[1]]], dtype=self.np.float32)      
            return new_points, status, error 


        def run(self, old_points, new_frame, old_frame):
            self.new_frame = new_frame

            if self.face_detected == False or self.frame_count > self.max_frame_face_track:
                self.face_detected = False
                self.frame_count = 0
                new_points, status, error = self.face_detection(new_frame)

                return new_points, status, error
            
            else:
                self.frame_count += 1
                new_points, status, error = self.optical_flow_tracking(old_frame, new_frame, old_points)
                
                return new_points, status, error