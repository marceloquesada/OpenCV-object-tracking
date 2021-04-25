import cv2
import numpy as np

canny_sens = 50
points = (0,0)
obj_center = (0,0)
obj_size = 20


def select_point(event, x, y, flags, params):
    global frame

    # Checa se o evento é um clique do botão esquerdo do mouse e, se sim, pega as coordenadas do clique e as passa para as variaveis
    if event == cv2.EVENT_LBUTTONDOWN:
        global points
        points = (x, y)
        get_obj_size(frame)
        point_is_selected = True


def get_obj_size(frame):
    global canny_sens, points, obj_center, obj_size

    edges = cv2.Canny(frame,0,canny_sens)

    edges = cv2.dilate(edges, None, iterations=2)
    
    obj_fill = edges.copy()
    frame_ed = frame.copy()
    h, w = edges.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    obj_fill = obj_fill.astype("uint8")
    cv2.floodFill(obj_fill, mask, points, 255)
    im_flood_fill_inv = cv2.bitwise_not(obj_fill)
    img_out = edges | im_flood_fill_inv
    img_out = cv2.bitwise_not(img_out)
    
    _,cnts, _ = cv2.findContours(img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for count in cnts:
        (rect_x, rect_y, rect_w, rect_h) = cv2.boundingRect(count)
        
        obj_center = (round(rect_x + (rect_w/2)), round(rect_y + (rect_h/2)))
        
        cv2.circle(frame_ed, obj_center, 2, (255, 0, 0), -1)
        
        obj_size = min(rect_x, rect_h)

        # desenha o retângulo
        cv2.rectangle(frame_ed, ((rect_x), (rect_y)),
        ((rect_x + rect_w), (rect_y + rect_h)), (0, 255, 0), 2)


    cv2.imshow("obj detection", frame_ed)
    cv2.imshow("obj", img_out)


def canny_sens_change(value):
    global canny_sens
    
    canny_sens = value
    

cap = cv2.VideoCapture(0)
cv2.namedWindow("Principal")
cv2.setMouseCallback("Principal", select_point)
cv2.createTrackbar("sensibilidade da detecção", "Principal", 1,400, canny_sens_change)

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 0)
    
    cv2.imshow("Principal", frame)
    
    if cv2.waitKey(1) == 27:
        break    
