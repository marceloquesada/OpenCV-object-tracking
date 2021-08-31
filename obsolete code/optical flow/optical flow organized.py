class object_tracking:

    cv2 = __import__("cv2")
    math = __import__("math")
    np = __import__("numpy")
    plt = __import__("matplotlib.pyplot")
    Servo_motor = __import__("servo_gpio.Servo_motor")


    rounding = 2 # Número de casas que o código arredondará as coordenadas passadas ao servo
    dpoints_threshold = 0.05 # Distância em pixels abaixo da qual o programa ignorará qualquer movimento
    pos_threshold = 30 # Distância em pixeis do ponto até o centro da imagem abaixo da qual o programa não tentará centralizar o objeto
    fov_horizontal = 361 # Campo de visão horizontal da camera(em graus)
    fov_vertical = 261 # Campo de visão vertical da camera(em graus)
    e_slow = 8 # número pelo qual os movimentos lentos serão divididos
    e_fast = 0.1 # número pelo qual os movimentos rápidos serão divididos
    status = 0 # Status de rastreamento  (1 - ativo / 0 - não ativo)
    size = (640, 480) # Resolução do frame
    canny_sensilibility = 30 # Sensibilidade da detecção de bordas
    start_position = (500, 500)

    # Variáveis técnicas:
    max_list_size = 3 # Numero máximo de object points que o programa guardará, SETAR PARA UM NÚMERO MAIOR QUE 2

    # Variáveis estáticas(NÃO ALTERAR)
    points = ()
    



    def __init__(self, servoH_port, servoV_port, start_position=(500, 500)):
        if servoH_port is None or servoV_port is None:
            self.useTracking = False
        else:
            self.servo_horizontal =self.Servo_motor(servoH_port)
            self.servo_vertical = self.Servo_motor(servoV_port)

            self.start_position = start_position

            self.servo_horizontal.set_angle(self.start_position[0])
            self.servo_vertical.set_angle(self.start_position[1])


    def obj_itentifier(self, frame):
        cv2 = self.cv2
        np = self.np
        edges = cv2.Canny(frame,0,self.canny_sensilibility)
        edges = cv2.dilate(edges, None, iterations=2)

        obj_fill = edges.copy()
        frame_edited = frame.copy()
        h, w = edges.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        obj_fill = obj_fill.astype("uint8")
        cv2.floodFill(obj_fill, mask, point, 255)
        im_flood_fill_inv = cv2.bitwise_not(obj_fill)
        img_out = img_out = cv2.bitwise_not(edges | im_flood_fill_inv)

        _,cnts, _ = cv2.findContours(img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts)== 0:
            return point

        else:
            for count in cnts:
                (rect_x, rect_y, rect_w, rect_h) = cv2.boundingRect(count)

                obj_center = (round(rect_x + (rect_w/2)), round(rect_y + (rect_h/2)))

                cv2.circle(frame_edited, obj_center, 2, (255, 0, 0), -1)

                obj_size = (rect_w, rect_h)

                # desenha o retângulo
                cv2.rectangle(frame_edited, ((rect_x), (rect_y)),
                ((rect_x + rect_w), (rect_y + rect_h)), (0, 255, 0), 2)

            N_PIXEL = obj_size

            return obj_center, obj_size


    def servo_communication(self, new_points, old_points):
        width = self.size[0]
        height = self.size[1]

        if abs(new_points[0] - old_points[0]) > self.dpoints_threshold and abs(new_points[1] - old_points[1]) > self.dpoints_threshold and self.status == 1:

            img_center = (int(width / 2), int(height / 2))
            pos_to_center = (img_center[0] - new_points[0], img_center[1] - new_points[1])

            if abs(pos_to_center[0]) > self.pos_threshold or abs(pos_to_center[1]) > self.pos_threshold:

                x_to_center = ((pos_to_center[0] * self.fov_horizontal) / width)
                y_to_center = ((pos_to_center[1] * self.fov_vertical) / height)

                e_x = self.dynamic_divider(self.e_slow, self.e_fast, x_to_center, width/2)
                e_y = self.dynamic_divider(self.e_slow, self.e_fast, y_to_center, height/2)

                servo_h_angle = self.servo_horizontal.get_angle() - round(x_to_center/e_x, self.rounding)
                servo_v_angle = self.servo_vertical.get_angle() - round(y_to_center/e_y, self.rounding)
                if servo_h_angle > 999: servo_h_angle = 999
                if servo_v_angle > 999: servo_v_angle = 999
                if servo_h_angle < 0: servo_h_angle = 0
                if servo_v_angle < 0: servo_v_angle = 0

                self.servo_horizontal.set_angle(servo_h_angle)
                self.servo_vertical.set_angle(servo_v_angle)

                print("Comando:", "X:", str(servo_h_angle), ", Y:", str(servo_v_angle))


    def dynamic_divider(self, ef, ei, deltap, pmax):
        deltap = abs(deltap)
        e = ((ef - ei)/(pmax**2))*((deltap - pmax)**2) + ei
        return e


    class optical_flow:
        cv2 = __import__("cv2")

        tracking_area = (100,100) # Quantidade de pixeis calculados ao redor do pixel central
        max_level = 2 # Nível de pirâmide usado
        epsilon = 0.05 # Valores menores de epsilon deixam o programa mais rápido, porém menos preciso
        of_iterations = 5 # Número de iterações do optical flow, quanto mais iteracoes, mais preciso é a detecção, porém roda mais devagar
        lk_params = dict( winSize =(tracking_area[0], tracking_area[1]),
         maxLevel=max_level, criteria=( cv2.TERM_CRITERIA_EPS | super.cv2.TERM_CRITERIA_COUNT, of_iterations, epsilon))

            
        def __init__(self):
            pass


        def run(self):
            #Calcula as novas coordenadas do objeto, armazenadas em new_points
            new_points, status, error = self.cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **self.lk_params)

            #Converte o status para 1 ou 0,dependendo se um objeto está sendo rastreado ou não
            status = status.ravel()[0]

            return new_points


    def update(self, frame):
        if len(self.points) >= self.max_list_size:
            self.max_list_size = self.max_list_size[:1]

        
    def start(self, initial_point):
        self.initial_point = initial_point

        print("Starting object tracking")
