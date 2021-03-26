import cv2
import numpy as np
from servo_gpio import Servo_motor

#definição dos hiperparametros
N_PIXEL = 100 # Quantidade de pixeis calculados ao redor do pixel central
MAX_LEVEL = 4 # Nível de pirâmide usado
EPSILON = 0.06 # Valores menores de epsilon deixam o programa mais rápido, porém menos preciso
OF_ITERATIONS = 10 # Número de iterações do optical flow, quanto mais iteracoes, mais preciso é a detecção, porém roda mais devagar
ROUNDING = 2 # Número casas que o código arredondará as coordenadas passadas ao servo
SERVO_THRESHOLD = 0.1
FOV_HORIZONTAL = 361 # Campo de visão horizontal da camera(em graus)
FOV_VERTICAL = 261 # Campo de visão vertical da camera(em graus)
E_MAX = 8
E_MIN = 0.7

# Definição de todas as váriaveis nulas
point = ()
point_is_selected = False
old_points = np.array([[]])
status = 0

servo_h = Servo_motor(20)
servo_v = Servo_motor(21)

# Inicializa os parâmetros para o optical flow
lk_params = dict( winSize =(N_PIXEL, N_PIXEL),
                  maxLevel=MAX_LEVEL,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, OF_ITERATIONS, EPSILON))


# Essa função é chamada quando o opencv detecta um evento no mouse, e passa as coordenadas do clique para a váriavel ponto
def select_point(event, x, y, flags, params):
    global point, point_is_selected, old_points

    # Checa se o evento é um clique do botão esquerdo do mouse e, se sim, pega as coordenadas do clique e as passa para as variaveis
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_is_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)
    
    
def n_pixel_change(value):
    global N_PIXEL
    print(value)
    N_PIXEL = value
    
    
def e_change(value):
    global E_MAX
    print(value)
    E_MAX = value


def servo_communication(new_points, old_points):
    if abs(new_points[0] - old_points[0]) > SERVO_THRESHOLD and abs(new_points[1] - old_points[1]) > SERVO_THRESHOLD and status == 1:

        img_center = (int(width / 2), int(height / 2))
        pos_to_center = (img_center[0] - new_points[0], img_center[1] - new_points[1])
        
        if abs(pos_to_center[0]) > 25 or abs(pos_to_center[1]) > 25:

            x_servo = ((pos_to_center[0] * FOV_HORIZONTAL) / width)
            y_servo = ((pos_to_center[1] * FOV_VERTICAL) / height)
            
            e_x = ((E_MAX - E_MIN)/width)*pos_to_center[0] + (E_MAX)
            e_y = ((E_MAX - E_MIN)/height)*pos_to_center[1] + (E_MAX)

            servo_h_angle = servo_h.get_angle() - round(x_servo/e_x, ROUNDING)
            servo_v_angle = servo_v.get_angle() - round(y_servo/e_y, ROUNDING)
            if servo_h_angle > 1000: servo_h_angle = 1000
            if servo_v_angle > 1000: servo_v_angle = 1000

            servo_h.set_angle(servo_h_angle)
            servo_v.set_angle(servo_v_angle)
            
            print(servo_h_angle, servo_v_angle)


def add_status_text(img):
    # Mostra na parte de cima da janela o status de rastreamento
    global point_is_selected
    if point_is_selected == 0:
        status_text = "Rastreamento nao ativo, clique no objeto"
        color_text = (0, 0, 200)
        font_scale = 0.7
    elif status == 1:
        status_text = "Rastreamento ativo!"
        color_text = (0, 200, 0)
        font_scale = 1.1
    else:
        status_text = "Rastreamento perdido"
        color_text = (0, 0, 200)
        font_scale = 1.3
    frame = cv2.putText(img, text=status_text, org=(80, 40), fontFace=cv2.QT_FONT_NORMAL, fontScale=font_scale,
                        color=color_text)


# Inicia a gravação de vídeo
cap = cv2.VideoCapture(0)

servo_h.set_angle(500)
servo_v.set_angle(500)

# Lê um frame para servir como o primeiro frame para ser comparado no optical flow
ret, frame = cap.read()
frame = cv2.flip(frame, 0)
height, width, chennels = frame.shape
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Cria janela principal e seta a função select_point para ser chamadaema qualquer evento do mouse
cv2.namedWindow("Principal")
cv2.setMouseCallback("Principal", select_point)
cv2.createTrackbar('N de pixeis rastreados', "Principal", 100, 300, n_pixel_change)
cv2.createTrackbar('Sensibilidade do rastreamento', "Principal", 8, 25, e_change)


while True:
    #Lê o frame da webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    
    #converte para preto e branco,para ser usado no opticalflow
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #O if roda apenas se um ponto para rastreamento foi selecionado
    if point_is_selected:

        #Calcula as novas coordenadas do objeto, armazenadas em new_points
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)

        #Converte o status para 1 ou 0,dependendo se um objeto está sendo rastreado ou não, e printa informações para debugging no console
        status = status.ravel()[0]

        #extrai as coordenadas do pontos para serem usadas nos próximos comandos
        xnew, ynew = new_points.ravel()
        xold, yold = old_points.ravel()

        # Só manda as coordenadas para os servos se as coordenadas mudaram, assim evitando sobrecarregar os servos

        servo_communication((xnew, ynew), (xold, yold))

        #Necessário para manter o loop, passando no próximo frame o new_points como old_points
        old_points = new_points

        #desenha uma linha entre o ponto antigo e ponto novo
        cv2.line(frame, (xnew, ynew), (xold, yold), (255, 0, 0), 4, -1)

        #Desenha um ponto na posição atual do objeto
        cv2.circle(frame, (xnew, ynew), 7, (170, 0, 170), 5)
        
        cv2.rectangle(frame, (int(xnew - N_PIXEL/2), int(ynew - N_PIXEL/2)), (int(xnew + N_PIXEL/2), int(ynew + N_PIXEL/2)), (0, 200, 0))


    #Seta old_frame como new_frame para a próxima iteração
    old_gray = new_gray.copy()

    add_status_text(frame)

    #Mostra a janela
    cv2.imshow("Principal", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
