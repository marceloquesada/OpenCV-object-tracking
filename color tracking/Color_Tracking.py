import cv2
import numpy as np

#os valores usados foram otimizados pra iluminação aqui e o objeto usado foi um post-it verde claro

#Lower e upper se referem a faixa verde na escala HSV
lower = np.array([32,35,0])
upper = np.array([83,255,255])

#iterações para remoção de ruído
iterat = 1

#threshold que determina o quão grande o contorno deve ser para ser desenhado
cnt_area_threshold = 600

#cria captura de vídeo
cam_cap = cv2.VideoCapture(0)

while True:
    #lê o frame atual e salva para "frame"
    ret, frame = cam_cap.read()

    #converte a imagem para HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    range = cv2.inRange(hsv,lower,upper)

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
    cnts, hierarchy = cv2.findContours(range, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #passa por cada contorno e se a área dele for maior que o threshold desenha um retângulo em volta do contorno e um circulo no centro
    for count in cnts:
        if cv2.contourArea(count) > cnt_area_threshold:
            (x, y, w, h) = cv2.boundingRect(count)

            #faz com que o retângulo seja 20% maior que a imagem
            rect_threshold_x = w // 5
            rect_threshold_y = h // 5

            #gera o ponto no centro do retângulo
            circle_x = w // 2
            circle_y = h // 2

            radius = 5

            #desenha o retângulo
            cv2.rectangle(cnt_frame, ((x-rect_threshold_x) , (y-rect_threshold_y)), ((x + w + rect_threshold_x) , (y + h + rect_threshold_y)), (0,255,0), 2)

            #desenha o circulo
            cv2.circle(cnt_frame, (x+circle_x,y+circle_y), radius, (255,0,0), -1)


    #mostra as imagens na tela, "q" para fechar as guias
    cv2.imshow("original", frame)
    cv2.imshow("mask", range)
    cv2.imshow("Color Tracking", cnt_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break