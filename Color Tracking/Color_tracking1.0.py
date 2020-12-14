import cv2
import numpy as np
import serial

#valores de campo de visão da câmera
angle_fov_horizontal = 65
angle_fov_vertical = 47.0

#essa variável divide o valor passado ao servo,assim diminuido a velocidade,como também overshooting *TORNAR DINAMICO*
speed_divider = 3.6

# os valores usados foram otimizados pra iluminação do local testado e o objeto usado foi um post-it verde claro

# Lower e upper se referem a faixa verde na escala HSV
lower = np.array([32, 38, 0])
upper = np.array([83, 255, 255])

#define a porta USB em que o arduino está conectado
arduino_port = "COM3"

# iterações para remoção de ruído
iterat = 1

#inicializa a lista de contornos
cnt_areas = []

# threshold que determina o quão grande o contorno deve ser para ser desenhado
cnt_area_threshold = 600

#inicializa a porta do arduino
arduino_serial = serial.Serial(arduino_port, 9600)

# cria captura de vídeo
cam_cap = cv2.VideoCapture(1)
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 480))


#divide os valores por spped_divider e os passa ao arduino no formato: x(angulo_x)y(angulo_y) exemplo:x10y20
def serial_communication(servo_horizontal, servo_vertical):
    data = "x" + str(servo_horizontal/speed_divider) + "y" + str(-servo_vertical/speed_divider)
    print(data)
    arduino_serial.write(data.encode())


#calcula o angulo ue o servo deve estar usando a equação: angulo_servo = delta_pos(x ou y) * fov / altura ou largura do frame
#essa equação foi derivada de uma regra de tres e os valores de fov foram calculados usando uma trena
def calculate_servo_angle(pos_to_center):
    angle_servo_horizontal = round((pos_to_center[0] * angle_fov_horizontal)/img_width,5)
    angle_servo_vertical = round((pos_to_center[1] * angle_fov_vertical) / img_height,5)
    
    serial_communication(angle_servo_horizontal,angle_servo_vertical)


while True:
    # lê o frame atual e salva para "frame"
    ret, frame = cam_cap.read()

    # converte a imagem para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_range = cv2.inRange(hsv, lower, upper)

    # cria 2 iterações de erosão e dilatação para limpar ruído, feito em while pois funciona melhor do que as iterações nativas
    i = 0
    while i < 2:
        color_range = cv2.erode(color_range, None, iterations=iterat * 2)
        color_range = cv2.dilate(color_range, None, iterations=iterat)
        i += 1
    color_range = cv2.dilate(color_range, None, iterations=4)

    # cria cópia da imagem para desenhar contornos
    cnt_frame = frame.copy()

    # acha todos os contornos na imagem
    cnts, hierarchy = cv2.findContours(color_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #extrai propriedades da imagem
    img_height, img_width, channels = cnt_frame.shape

    #desenha um circulo no centro da imagem para referência
    cv2.circle(cnt_frame, (int(img_width / 2), int(img_height / 2)), 5, (0, 0, 255), -1)

    cnt_areas = []

    for cnt in cnts:
        cnt_areas.append(cv2.contourArea(cnt))

    # passa por cada contorno e se a área dele for maior que o threshold desenha um retângulo em volta do contorno e um circulo no centro
    for count in cnts:
        if cv2.contourArea(count) > cnt_area_threshold:
            (rect_x, rect_y, rect_w, rect_h) = cv2.boundingRect(count)

            # faz com que o retângulo seja 20% maior que a imagem
            rect_threshold_x = rect_w // 5
            rect_threshold_y = rect_h // 5

            # gera o ponto no centro do retângulo
            circle_x = rect_w // 2
            circle_y = rect_h // 2

            radius = 5

            # desenha o retângulo
            cv2.rectangle(cnt_frame, ((rect_x - rect_threshold_x), (rect_y - rect_threshold_y)),
                          ((rect_x + rect_w + rect_threshold_x), (rect_y + rect_h + rect_threshold_y)), (0, 255, 0), 2)

            # desenha o circulo no centro do objeto
            cv2.circle(cnt_frame, (rect_x + circle_x, rect_y + circle_y), radius, (255, 0, 0), -1)

            x = rect_x + circle_x
            y = rect_y + circle_y
            pos = (x,y)

            img_center = (int(img_width / 2), int(img_height / 2))
            pos_to_center = (img_center[0] - x, img_center[1] - y)

            #desenha distância entre o centro do objeto e o centro da imagem
            cv2.line(cnt_frame, (x,y), (img_center[0], img_center[1]), (150 , 0 , 50), 2)
            sorted_areas = cnt_areas.sort()
            
            #um problema nesse código é que quando a camera detecta dois objetos essa função é passada duas vezes,
            #variando o servo entre duas posições,essa foi minha tentativa de consertar(não funciona)
            try:
                if sorted_areas[0] == cv2.contourArea(count):
                    calculate_servo_angle(pos_to_center=pos_to_center)
            except:
                calculate_servo_angle(pos_to_center=pos_to_center)

    # mostra as imagens na tela, "q" para fechar as guias
    cv2.imshow("original", frame)
    cv2.imshow("mask", color_range)
    cv2.imshow("Color Tracking", cnt_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#libera captura de tela
cam_cap.release()