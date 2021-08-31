from numpy import select


class optical_flow_ROI:

        cv2 = __import__("cv2")
        np = __import__("numpy")
        
        tracking_area = (100,100) # Quantidade de pixeis rastreados ao redor do pixel central
        max_level = 2 # Nível de pirâmide usado
        epsilon = 0.05 # Valores menores de epsilon deixam o programa mais rápido, porém menos preciso
        iterations = 5 # Número de iterações do optical flow, quanto mais iteracoes, mais preciso é a detecção, porém roda mais devagar

        # ROI selection variables
        roi_start_event = cv2.EVENT_LBUTTONDOWN
        window_name = "Roi select"


        #Technical variables(avoid changing)
        roi_selected = False
        event_detected = False

        point_ROI = None

        points = []

        
        
        def __init__(self, window_name = window_name, tracking_area = tracking_area, max_level = max_level, epsilon = epsilon, iterations = iterations):
            self.tracking_area = tracking_area
            self.max_level = max_level
            self.epsilon = epsilon
            self.iterations = iterations
            self.window_name = window_name

            self.cv2.namedWindow(window_name)
            self.cv2.setMouseCallback(window_name, self.event_handler)
        

        def gray_scale(self, frame):
            frame_gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)

            return frame_gray


        def event_handler(self, event, x, y, flags, params):
            if event == self.roi_start_event:
                self.event_detected = True
                self.selected_points = (x, y)
            
                roi = self.cv2.selectROI("ROI", self.new_frame)
                self.cv2.destroyWindow("ROI")
                self.roi_selected = True

                self.tracking_area = (roi[2], roi[3])

                point_roi = ((roi[0] + (roi[2]/2)) , (roi[1] + (roi[3]/2)))
                self.point_ROI = self.np.array([[ point_roi ]], dtype=self.np.float32)
            


        def run(self, old_points, new_frame, old_frame):
            self.new_frame = new_frame

            if self.roi_selected:
                self.lk_params = dict( winSize =(self.tracking_area[0], self.tracking_area[1]),
                maxLevel=self.max_level, criteria=( self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT, self.iterations, self.epsilon))

                if self.point_ROI is not None:
                    old_points = self.point_ROI
                    self.point_ROI = None
                #Calcula as novas coordenadas do objeto, armazenadas em new_points
                new_gray = self.gray_scale(new_frame)
                old_gray = self.gray_scale(old_frame)

                new_points, status, error = self.cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **self.lk_params)

                #Converte o status para 1 ou 0,dependendo se um objeto está sendo rastreado ou não
                status = status.ravel()[0]
                
                return new_points, status, error
            else:
                return old_points, 0, 0
