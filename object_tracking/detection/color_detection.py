import cv2

class color_detection:
    np = __import__("numpy")

    lower = np.array([32,35,40])
    upper = np.array([83,255,255])
    iterations = 2
    cnt_area_threshold = 0
    tracking_area = (0,0)

            
    def __init__(self, lower_hsv=lower, upper_hsv=upper):
        self.lower = lower_hsv
        self.upper = upper_hsv


    def run(self, old_points, new_frame, old_frame):
        hsv = cv2.cvtColor(new_frame,cv2.COLOR_BGR2HSV)
        inRange = cv2.inRange(hsv,self.lower,self.upper)

        for i in range(0,2):
            inRange = cv2.erode(inRange, None, iterations=self.iterations*2)
            inRange = cv2.dilate(inRange, None, iterations=self.iterations)

        #acha todos os contornos na imagem
        _, cnts, _ = cv2.findContours(inRange, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        countours = 0
        status = 0

        #passa por cada contorno e se a área dele for maior que o threshold desenha um retângulo em volta do contorno e um circulo no centro
        for count in cnts:
            if cv2.contourArea(count) > self.cnt_area_threshold:
                countours += 1
                status = 1
                (x, y, w, h) = cv2.boundingRect(count)
                self.tracking_area = (w, h)
                new_points = ((x + (w/2)), (y + h/2))
        
        if countours == 0:
            error = 0
            return old_points, status, error
        else:
            error = 0
            new_points = self.np.array([[new_points]], dtype=self.np.float32)

            return new_points, status, error