import math
import cv2
import serial
import matplotlib.pyplot as plt
import numpy as np
from servo_gpio import Servo_motor


#definição dos hiperparametros
ROUNDING = 3 # Número casas que o código arredondará as coordenadas passadas ao servo
POINTS_THRESHOLD = 0.05 # Distância em pixels abaixo da qual o programa ignorará qualquer movimento
POS_TO_CENTER_THRESHOLD = 30 # Distância em pixeis do ponto até o centro da imagem abaixo da qual o programa não tentará centralizar o objeto
FOV_HORIZONTAL = 361 # Campo de visão horizontal da camera(em graus)
FOV_VERTICAL = 261 # Campo de visão vertical da camera(em graus)
E_SLOW = 12 # número pelo qual os movimentos lentos serão divididos
E_FAST = 0.2 # número pelo qual os movimentos rápidos serão divididos
lower = np.array([32, 60, 60])
upper = np.array([83, 255, 255])
iterat = 1
cnt_area_threshold = 600

# Definição de todas as váriaveis nulas
point = ()
obj_is_selected = False
old_points = (0,0)
status = 0
started_experiment = 0


graph = "convexo"
vel = "1"
ite = "2"


ser = serial.Serial(
        port='/dev/ttyACM0',
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
        file = open(("log" + graph + vel + "_" + ite + ".txt"), 'w+')
        file.write("distancia até centro em x: " + str(self.pos_x) + "\n")
        file.write("distancia até centro em Y: " + str(self.pos_y) + "\n")
        file.write("distancia total até o centro: " + str(self.dist) + "\n")
        file.close()

    def plot(self):
        tempo = np.arange(0, len(self.dist))
        plt.xlabel('time')
        plt.ylabel('distance to center')
        fig = plt.figure()
        plt.plot(tempo, self.dist)
        fig.savefig('graficos/grafico ' + graph + '/cpc' + vel + "_" + ite + '.png', dpi=fig.dpi)
        
        tempo = np.arange(0, len(self.e))
        plt.xlabel('time')
        plt.ylabel('value of e')
        fig = plt.figure()
        plt.plot(tempo, self.e)
        fig.savefig('graficos/grafico ' + graph + '/e' + vel + "_" + ite + '.png', dpi=fig.dpi)

        tempo = np.arange(0, len(self.pos_x))
        plt.xlabel('time')
        plt.ylabel('distance to center x')
        fig = plt.figure()
        plt.plot(tempo, self.pos_x)
        fig.savefig('graficos/grafico ' + graph + '/cpcx' + vel + "_" + ite + '.png', dpi=fig.dpi)

        tempo = np.arange(0, len(self.pos_y))
        plt.xlabel('time')
        plt.ylabel('distance to center y')
        fig = plt.figure()
        plt.plot(tempo, self.pos_y)
        fig.savefig('graficos/grafico ' + graph + '/cpcy' + vel + "_" + ite + '.png', dpi=fig.dpi)


#Inicialização dos servos
servo_h = Servo_motor(20)
servo_v = Servo_motor(21)

pos_to_center_data = data_logging()


def color_detection(old_gray, frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    range = cv2.inRange(hsv,lower,upper)

    x = 0
    y = 0
    w = 0
    h = 0

    #cria 2 iterações de erosão e dilatação para limpar ruído, feito em while pois funciona melhor do que as iterações nativas
    i = 0
    while i < 2:
        range = cv2.erode(range, None, iterations=iterat*2)
        range = cv2.dilate(range, None, iterations=iterat)
        i += 1
    range = cv2.dilate(range, None, iterations=4)

    #cria cópia da imagem para desenhar contornos
    cnt_frame = frame.copy()

    #acha todos os contornos na imagem
    _,cnts,_ = cv2.findContours(range, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #passa por cada contorno e se a área dele for maior que o threshold desenha um retângulo em volta do contorno e um circulo no centro
    for count in cnts:
        if cv2.contourArea(count) > cnt_area_threshold:
            (x, y, w, h) = cv2.boundingRect(count)
    
    new_points = ((x + (w/2)), (y + (h/2)))

    return new_points, (x, y, w, h)


def movement_divider(ef, ei, deltap, pmax):
    deltap = abs(deltap)
    e = ((ef - ei)/(pmax**2))*((deltap - pmax)**2) + ei
    return e


# Calcula o angulo que deve ser passado aos servos(0 - 1000), considerando a velocidade do movimento e o ângulo que o motor já se encontra
def servo_communication(new_points, old_points):
    global started_experiment
    
    if started_experiment == False:
        ser.write((vel + ".").encode());
        started_experiment = True

    pos_to_center = (img_center[0] - new_points[0], img_center[1] - new_points[1])

    total_distance = ((((img_center[0] - new_points[0])**2) + ((img_center[1] - new_points[1])**2) )**0.5)

    pos_to_center_data.att(pos_to_center, total_distance)

    if abs(new_points[0] - old_points[0]) > POINTS_THRESHOLD and abs(new_points[1] - old_points[1]) > POINTS_THRESHOLD:

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
servo_h.set_angle(300)
servo_v.set_angle(500)

# Lê um frame para servir como o primeiro frame para ser comparado no optical flow
ret, frame = cap.read()
frame = cv2.flip(frame, 0)
height, width, chennels = frame.shape
img_center = (int(width / 2), int(height / 2))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Cria janela principal e seta a função select_point para ser chamada em qualquer evento do mouse
cv2.namedWindow("Principal")

# Cria trackbars para controlar manualmente E_SLOW, E_FAST, CANNY_EDGES e N_PIXEL

cv2.createTrackbar('E_SLOW', "Principal", E_SLOW, 25, eslow_change)
cv2.createTrackbar('E_FAST', "Principal", int(E_FAST*100), 100, efast_change)

while True:
    #Lê o frame da webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)

    #O if roda apenas se um ponto para rastreamento foi selecionado

    new_points, (x, y, w, h) = color_detection(old_gray, frame)

    #extrai as coordenadas do pontos para serem usadas nos próximos comandos
    xnew, ynew = new_points
    xold, yold = old_points

    # Só manda as coordenadas para os servos se as coordenadas mudaram, assim evitando sobrecarregar os servos

    servo_communication((xnew, ynew), (xold, yold))

    #Necessário para manter o loop, passando no próximo frame o new_points como old_points
    old_points = new_points

    #desenha uma linha entre o ponto antigo e ponto novo
    edited_frame = cv2.line(frame, (int(xnew), int(ynew)), (int(xold), int(yold)), (255, 0, 0), 4, -1)

    #Desenha um ponto na posição atual do objeto
    edited_frame = cv2.circle(frame, (int(xnew), int(ynew)), 7, (170, 0, 170), 5)

    edited_frame = cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0))

    #Mostra a janela
    cv2.imshow("Principal", frame)

    if cv2.waitKey(1) == 27:
        break

# Destrói as janelas e fecha a captura de vídeo quando o ESC é pressionado
cap.release()
cv2.destroyAllWindows()
pos_to_center_data.dump()
pos_to_center_data.plot()

