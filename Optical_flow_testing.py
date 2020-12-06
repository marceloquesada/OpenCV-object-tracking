import cv2
import numpy as np


#essa função é chamada quando o opencv detecta um evento no mouse, e passa as coordenadas do clique para a váriavel ponto
def select_point(event, x, y, flags, params):
    global point, point_is_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_is_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


#Essa parte determina em que janela que os cliques serão detectados e inicializa a detecção
cv2.namedWindow("Principal")
cv2.setMouseCallback("Principal", select_point)


#Inicia a gravação de vídeo
cap = cv2.VideoCapture(0)


#Inicializa os parâmetros para o optical flow
lk_params = dict( winSize =(40, 40),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Lê um frame para servir como o primeiro frame para ser comparado no optical flow
ret, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#Inicializa todas as váriaveis a serem usadas depois
point = ()
point_is_selected = False
old_points = np.array([[]])
reference_frame = np.zeros((480,640,3), np.uint8)
status = 0


while True:
    #Lê o frame da webcam
    ret, frame = cap.read()

    #converte para preto e branco,para ser usado no opticalflow
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #O if roda apenas se um ponto para rastreamento foi selecionado
    if point_is_selected:

        #Calcula as novas coordenadas do objeto, armazenadas em new_points
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)

        #Converte o status para 1 ou 0,dependendo se um objeto está sendo rastreado ou não, e printa infromações para debugging no console
        status = status.ravel()[0]
        print("Old point: ", old_points)
        print("New point: ", new_points)
        print("Status: ", status)
        print("Error: ", error)
        print("------------------------------------------------------------")

        #extrai as coordenadas do pontos para serem usadas nos próximos comandos
        xnew, ynew = new_points.ravel()
        xold, yold = old_points.ravel()

        #Necessário para manter o loop, passando no próximo frame o new_points como old_points
        old_points = new_points

        #desenha uma linha entre o ponto antigo e ponto novo
        cv2.line(reference_frame, (xnew, ynew), (xold, yold), (255, 0, 0), 4, -1)

        #Desenha um ponto na posição atual do objeto
        cv2.circle(frame, (xnew, ynew), 7, (170, 0, 170), 5)


    #Mostra na parte de cima da janela o status de rastreamento
    if status == 1:
        status_text = "Rastreamento ativo!"
        color_text = (0,255,0)
    else:
        reference_frame = np.zeros((480, 640, 3), np.uint8)
        status_text = "Rastreamento nao ativo ou perdido"
        color_text = (0,0,255)
    frame = cv2.putText(frame, text=status_text, org=(0, 40), fontFace=cv2.QT_FONT_NORMAL, fontScale=1, color=color_text)

    #Seta old_frame como new_frame para a próxima iteração
    old_gray = new_gray.copy()

    #Mescla o frame da webcam + ponto atual do objeto + status com o frame contendo as linhas de posições passadas
    show_frame = cv2.add(frame, reference_frame)

    #Mostra a janela
    cv2.imshow("Principal", show_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
