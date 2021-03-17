import cv2
import numpy as np


#definição dos hiperparametros
N_PIXEL = 40 # Quantidade de pixeis calculados ao redor do pixel central
MAX_LEVEL = 2 # Nível de pirâmide usado
EPSILON = 0.03 # Valores menores de epsilon deixam o programa mais rápido, porém menos preciso
OF_ITERATIONS = 10 # Número de iterações do optical flow, quanto mais iteracoes, mais preciso é a detecção, porém roda mais devagar
ROUNDING = 2 # Número casas que o código arredondará as coordenadas passadas ao servo
SERVO_THRESHOLD = 0.5


# Definição de todas as váriaveis nulas
point = ()
point_is_selected = False
old_points = np.array([[]])
status = 0


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


def servo_communication(new_points, old_points):
    if abs(new_points[0] - old_points[0]) > SERVO_THRESHOLD and abs(new_points[1] - old_points[1]) > SERVO_THRESHOLD and status == 1:
        x_servo = (180*new_points[0])/width
        y_servo = (180*new_points[1])/height
        if x_servo > 180: x_servo = 180
        if y_servo > 180: y_servo = 180

        cmm = "X" + str(round(x_servo, ROUNDING)) + "Y" + str(round(y_servo, ROUNDING))

        print(cmm)


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

# Lê um frame para servir como o primeiro frame para ser comparado no optical flow
ret, frame = cap.read()
height, width, chennels = frame.shape
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Cria janela principal e seta a função select_point para ser chamadaema qualquer evento do mouse
cv2.namedWindow("Principal")
cv2.setMouseCallback("Principal", select_point)


while True:
    #Lê o frame da webcam
    ret, frame = cap.read()

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


    #Seta old_frame como new_frame para a próxima iteração
    old_gray = new_gray.copy()

    add_status_text(frame)

    #Mostra a janela
    cv2.imshow("Principal", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
