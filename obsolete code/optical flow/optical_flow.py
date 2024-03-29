import math
import cv2
import serial
import matplotlib.pyplot as plt
import numpy as np
from RpyServo import Servo_motor


#definição dos hiperparametros
N_PIXEL = 100 # Quantidade de pixeis calculados ao redor do pixel central
MAX_LEVEL = 2 # Nível de pirâmide usado
EPSILON = 0.05 # Valores menores de epsilon deixam o programa mais rápido, porém menos preciso
OF_ITERATIONS = 5 # Número de iterações do optical flow, quanto mais iteracoes, mais preciso é a detecção, porém roda mais devagar
ROUNDING = 2 # Número casas que o código arredondará as coordenadas passadas ao servo
POINTS_THRESHOLD = 0.05 # Distância em pixels abaixo da qual o programa ignorará qualquer movimento
POS_TO_CENTER_THRESHOLD = 30 # Distância em pixeis do ponto até o centro da imagem abaixo da qual o programa não tentará centralizar o objeto
FOV_HORIZONTAL = 361 # Campo de visão horizontal da camera(em graus)
FOV_VERTICAL = 261 # Campo de visão vertical da camera(em graus)
E_SLOW = 25 # número pelo qual os movimentos lentos serão divididos
E_FAST = 0.7 # número pelo qual os movimentos rápidos serão divididos
CANNY_SENS = 30 # Sensibilidade da detecção de bordas


# Definição de todas as váriaveis nulas
point = ()
obj_is_selected = False
old_points = np.array([[]])
status = 0
started_experiment = 0

ser = serial.Serial(
        port='/dev/ttyACM0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 9600
)

class data_logging:
    def __init__(self):
        self.pos_x = []
        self.pos_y = []
        self.dist = []
        self.e = []

    def att(self, points, dist):
        self.pos_x.append(points[0])
        self.pos_y.append(points[1])
        self.dist.append(dist)

    def dump(self):
        file = open("log.txt", 'w+')
        file.write("distancia até centro em x: " + str(self.pos_x) + "\n")
        file.write("distancia até centro em Y: " + str(self.pos_y) + "\n")
        file.write("distancia total até o centro: " + str(self.dist) + "\n")
        file.close()

    def plot(self):
        tempo = np.arange(0, len(self.dist))
        plt.xlabel('time')
        plt.ylabel('distance to center')
        plt.plot(tempo, self.dist)
        plt.show()
        plt.savefig('distance_to_center.png')
        
        tempo = np.arange(0, len(self.e))
        plt.xlabel('time')
        plt.ylabel('value of e')
        plt.plot(tempo, self.e)
        plt.show()
        plt.savefig('e.png')

        tempo = np.arange(0, len(self.pos_x))
        plt.xlabel('time')
        plt.ylabel('distance to center x')
        plt.plot(tempo, self.pos_x)
        plt.show()
        plt.savefig('distance_to_center_x.png')

        tempo = np.arange(0, len(self.pos_y))
        plt.xlabel('time')
        plt.ylabel('distance to center y')
        plt.plot(tempo, self.pos_y)
        plt.show()
        plt.savefig('distance_to_center_y.png')



#Inicialização dos servos
servo_h = Servo_motor(20)
servo_v = Servo_motor(21)

pos_to_center_data = data_logging()

# Essa função é chamada quando o opencv detecta um evento no mouse, e passa as coordenadas do clique para a váriavel ponto
def select_point(event, x, y, flags, params):
    global point, obj_is_selected, old_points, frame

    # Checa se o evento é um clique do botão esquerdo do mouse e, se sim, pega as coordenadas do clique e as passa para as variaveis
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        x_obj, y_obj = x, y
        obj_is_selected = True
        old_points = np.array([[x_obj, y_obj]], dtype=np.float32)


# Essa função roda um algoritmo de detecção de bordas, pega o ponto que o usuário clicou usando a função floodfill gera uma mascara do objeto
# Depois, encontra o bounding rectangle e encontra o centro do objeto,
# passando a cordenada mínima do retangulo com N_PIXEL e ponto no centro do objeto como o novo point
def get_obj_size_and_center(frame):
    global CANNY_SENS, point, N_PIXEL, obj_size

    edges = cv2.Canny(frame,0,CANNY_SENS)

    edges = cv2.dilate(edges, None, iterations=2)

    obj_fill = edges.copy()
    frame_ed = frame.copy()
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

            cv2.circle(frame_ed, obj_center, 2, (255, 0, 0), -1)

            obj_size = min(rect_x, rect_h)

            # desenha o retângulo
            cv2.rectangle(frame_ed, ((rect_x), (rect_y)),
            ((rect_x + rect_w), (rect_y + rect_h)), (0, 255, 0), 2)

        N_PIXEL = obj_size

        return obj_center


def optical_flow_obj_detection(old_gray, new_gray, old_points):
    global status

    # Inicializa os parâmetros para o optical flow
    lk_params = dict( winSize =(N_PIXEL, N_PIXEL),
                  maxLevel=MAX_LEVEL,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, OF_ITERATIONS, EPSILON))

    #Calcula as novas coordenadas do objeto, armazenadas em new_points
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)

    #Converte o status para 1 ou 0,dependendo se um objeto está sendo rastreado ou não
    status = status.ravel()[0]

    return new_points


def movement_divider(ef, ei, deltap, pmax):
    deltap = abs(deltap)
    e = ((ef - ei)/(pmax**2))*((deltap - pmax)**2) + ei
    return e


# Calcula o angulo que deve ser passado aos servos(0 - 1000), considerando a velocidade do movimento e o ângulo que o motor já se encontra
def servo_communication(new_points, old_points):
    global started_experiment
    
    if started_experiment == False:
        ser.write("start\n".encode());
        started_experiment = 1

    pos_to_center = (img_center[0] - new_points[0], img_center[1] - new_points[1])

    total_distance = ((((img_center[0] - new_points[0])**2) + ((img_center[1] - new_points[1])**2) )**0.5)

    pos_to_center_data.att(pos_to_center, total_distance)

    if abs(new_points[0] - old_points[0]) > POINTS_THRESHOLD and abs(new_points[1] - old_points[1]) > POINTS_THRESHOLD and status == 1:

        if True:

            x_to_center = ((pos_to_center[0] * FOV_HORIZONTAL) / width)
            y_to_center = ((pos_to_center[1] * FOV_VERTICAL) / height)

            e_x = movement_divider(E_SLOW, E_FAST, x_to_center, width/2)
            e_y = movement_divider(E_SLOW, E_FAST, y_to_center, height/2)
            
            pos_to_center_data.e.append(e_x)

            servo_h_angle = servo_h.get_angle() - round(x_to_center/e_x, ROUNDING)
            servo_v_angle = servo_v.get_angle() - round(y_to_center/e_y, ROUNDING)
            if servo_h_angle > 999: servo_h_angle = 999
            if servo_v_angle > 999: servo_v_angle = 999
            if servo_h_angle < 0: servo_h_angle = 0
            if servo_v_angle < 0: servo_v_angle = 0

            servo_h.set_angle(servo_h_angle)
            servo_v.set_angle(servo_v_angle)

            print("Comando:", "X:", str(servo_h_angle), ", Y:", str(servo_v_angle))


# Mostra na parte de cima da janela o status de rastreamento
def add_status_text(img):
    global obj_is_selected
    if obj_is_selected == 0:
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
    edited_frame = cv2.putText(img, text=status_text, org=(80, 40), fontFace=cv2.QT_FONT_NORMAL, fontScale=font_scale,
                        color=color_text)


# As funções abaixo são chamadas quando os valores das trackbars são alterados, assim alterando o valor geral das variaveis associadas a elas
def n_pixel_change(value):
    global N_PIXEL
    print("área de rastreamento:", str(value))
    N_PIXEL = value


def eslow_change(value):
    global E_SLOW
    print("E_MIN:", str(value))
    E_MAX = value


def efast_change(value):
    global E_FAST
    value = value / 10
    print("E_FAST:", str(value))
    E_FAST = value


def canny_sens_change(value):
    global CANNY_SENS
    print("sensibilidade de detecção:", str(value))
    CANNY_SENS = value


def pos_to_center_change(value):
    global POS_TO_CENTER_THRESHOLD
    print("pos to center:", str(value))
    POS_TO_CENTER_THRESHOLD = value



# Inicia a gravação de vídeo
cap = cv2.VideoCapture(0)

# Seta o angulo inicial dos servos
servo_h.set_angle(500)
servo_v.set_angle(500)

# Lê um frame para servir como o primeiro frame para ser comparado no optical flow
ret, frame = cap.read()
frame = cv2.flip(frame, 0)
height, width, chennels = frame.shape
img_center = (int(width / 2), int(height / 2))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Cria janela principal e seta a função select_point para ser chamada em qualquer evento do mouse
cv2.namedWindow("Principal")
cv2.setMouseCallback("Principal", select_point)

# Cria trackbars para controlar manualmente E_SLOW, E_FAST, CANNY_EDGES e N_PIXEL

cv2.createTrackbar('N de pixeis rastreados', "Principal", N_PIXEL, 300, n_pixel_change)
cv2.createTrackbar('Threshold pos do centro', "Principal", POS_TO_CENTER_THRESHOLD, 50, pos_to_center_change)
cv2.createTrackbar('E_SLOW', "Principal", E_SLOW, 25, eslow_change)
cv2.createTrackbar('E_FAST', "Principal", int(E_FAST*100), 100, efast_change)
cv2.createTrackbar('Sensib. deteccao borda', "Principal", CANNY_SENS,400, canny_sens_change)


while True:
    #Lê o frame da webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)

    #converte para preto e branco,para ser usado no opticalflow
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #O if roda apenas se um ponto para rastreamento foi selecionado
    if obj_is_selected:

        new_points = optical_flow_obj_detection(old_gray, new_gray, old_points)

        #extrai as coordenadas do pontos para serem usadas nos próximos comandos
        xnew, ynew = new_points.ravel()
        xold, yold = old_points.ravel()

        # Só manda as coordenadas para os servos se as coordenadas mudaram, assim evitando sobrecarregar os servos

        servo_communication((xnew, ynew), (xold, yold))

        #Necessário para manter o loop, passando no próximo frame o new_points como old_points
        old_points = new_points

        #desenha uma linha entre o ponto antigo e ponto novo
        edited_frame = cv2.line(frame, (xnew, ynew), (xold, yold), (255, 0, 0), 4, -1)

        #Desenha um ponto na posição atual do objeto
        edited_frame = cv2.circle(frame, (xnew, ynew), 7, (170, 0, 170), 5)

        edited_frame = cv2.rectangle(frame, (int(xnew - N_PIXEL/2), int(ynew - N_PIXEL/2)), (int(xnew + N_PIXEL/2), int(ynew + N_PIXEL/2)), (0, 200, 0))


    #Seta old_frame como new_frame para a próxima iteração
    old_gray = new_gray.copy()

    add_status_text(frame)

    #Mostra a janela
    cv2.imshow("Principal", frame)

    if cv2.waitKey(1) == 27:
        break

# Destrói as janelas e fecha a captura de vídeo quando o ESC é pressionado
cap.release()
cv2.destroyAllWindows()
pos_to_center_data.dump()
pos_to_center_data.plot()

