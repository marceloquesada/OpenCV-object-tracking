import cv2
import numpy as np


## Definição dos hiperparâmetros

FOV_HORIZONTAL = 65 # Campo de visão horizontal da camera(em graus)
FOV_VERTICAL = 47.0 # Campo de visão vertical da camera(em graus)

# Lower e upper se referem a faixa verde na escala HSV(os valores usados foram otimizados pra iluminação aqui e o objeto usado foi um post-it verde claro)
LOWER_GREEN_SCALE = np.array([32, 38, 0])
UPPER_GREEN_SCALE = np.array([83, 255, 255])

NOISE_CANCEL_ITERAT = 1 # Iterações para remoção de ruído
CNT_AREA_THRESHOLD = 600 # Threshold que determina o quão grande o contorno deve ser para ser desenhado


# Definição do vetor contendo a area dos contornos para uso posterior
cnt_areas = []


# Cria captura de vídeo
cam_cap = cv2.VideoCapture(0)

# Calcula o angulo ue o servo deve estar usando a equação: angulo_servo = delta_pos(x ou y) * fov / altura ou largura do frame
# Essa equação foi derivada de uma regra de tres e os valores de fov foram calculados usando uma trena
# Para esses valores serem usados para servos temos que adicionar 90,já que os valores aceitos vão de 0 a 180


def serial_communication(servo_horizontal, servo_vertical):
    data = "x" + str(servo_horizontal+90) + "y" + str(servo_vertical+90)
    print(data)


def calculate_servo_angle(pos_to_center):
    angle_servo_horizontal = round((pos_to_center[0] * FOV_HORIZONTAL) / img_width, 5)
    angle_servo_vertical = round((pos_to_center[1] * FOV_VERTICAL) / img_height, 5)

    serial_communication(angle_servo_horizontal, angle_servo_vertical)


while True:
    # Lê o frame atual e salva para "frame"
    ret, frame = cam_cap.read()

    # Converte a imagem para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_range = cv2.inRange(hsv, LOWER_GREEN_SCALE, UPPER_GREEN_SCALE)

    # Cria 2 iterações de erosão e dilatação para limpar ruído, feito em while pois funciona melhor do que as iterações nativas
    for i in range(NOISE_CANCEL_ITERAT):
        color_range = cv2.erode(color_range, None, iterations=NOISE_CANCEL_ITERAT * 2)
        color_range = cv2.dilate(color_range, None, iterations=NOISE_CANCEL_ITERAT)
    color_range = cv2.dilate(color_range, None, iterations=4)

    # Cria cópia da imagem para desenhar contornos
    cnt_frame = frame.copy()

    # Acha todos os contornos na imagem
    cnts, hierarchy = cv2.findContours(color_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extrai propriedades da imagem
    img_height, img_width, channels = cnt_frame.shape

    # Desenha um circulo no centro da imagem para referência
    cv2.circle(cnt_frame, (int(img_width / 2), int(img_height / 2)), 5, (0, 0, 255), -1)

    cnt_areas = []

    for cnt in cnts:
        cnt_areas.append(cv2.contourArea(cnt))

    # Passa por cada contorno e se a área dele for maior que o threshold desenha um retângulo em volta do contorno e um circulo no centro
    for count in cnts:
        if cv2.contourArea(count) > CNT_AREA_THRESHOLD:
            (rect_x, rect_y, rect_w, rect_h) = cv2.boundingRect(count)

            # Faz com que o retângulo seja 20% maior que a imagem
            rect_threshold_x = rect_w // 5
            rect_threshold_y = rect_h // 5

            # Gera o ponto no centro do retângulo
            circle_x = rect_w // 2
            circle_y = rect_h // 2

            radius = 5

            # Desenha o retângulo
            cv2.rectangle(cnt_frame, ((rect_x - rect_threshold_x), (rect_y - rect_threshold_y)),
                          ((rect_x + rect_w + rect_threshold_x), (rect_y + rect_h + rect_threshold_y)), (0, 255, 0), 2)

            # Desenha o circulo no centro do objeto
            cv2.circle(cnt_frame, (rect_x + circle_x, rect_y + circle_y), radius, (255, 0, 0), -1)

            x = rect_x + circle_x
            y = rect_y + circle_y
            pos = (x,y)

            img_center = (int(img_width / 2), int(img_height / 2))
            pos_to_center = (img_center[0] - x, img_center[1] - y)

            # Desenha distância entre o centro do objeto e o centro da imagem
            cv2.line(cnt_frame, (x, y), (img_center[0], img_center[1]), (150, 0, 50), 2)
            sorted_areas = cnt_areas.sort()


    # Mostra as imagens na tela, "q" para fechar as guias
    cv2.imshow("original", frame)
    cv2.imshow("mask", color_range)
    cv2.imshow("Color Tracking", cnt_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_cap.release()
cv2.destroyAllWindows()
