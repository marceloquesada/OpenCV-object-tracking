import cv2
import numpy as np


def identify_object(self, frame, starting_point):
    edges = cv2.Canny(frame,0,self.canny_sensilibility)
    edges = cv2.dilate(edges, None, iterations=2)

    obj_fill = edges.copy()
    frame_edited = frame.copy()
    h, w = edges.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    obj_fill = obj_fill.astype("uint8")
    cv2.floodFill(obj_fill, mask, starting_point, 255)
    im_flood_fill_inv = cv2.bitwise_not(obj_fill)
    img_out = img_out = cv2.bitwise_not(edges | im_flood_fill_inv)

    _,cnts, _ = cv2.findContours(img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts)== 0:
        return starting_point, (0,0)

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
