from Constants import *
import cv2

def show_detections(image, boxes, names, distances, confidences, swap_to_RGB=True):
    for (x1, y1, x2, y2), name, (dist_x, dist_y), confidence in zip(boxes, names, distances, confidences):
        # Print confidence
        cv2.rectangle(image, (x1, y1), (x2, y2), color=BGR_BLUE, thickness=RECTANGLES_THICKNESS)
        txt = 'Conf: {conf}%'.format(conf=round(confidence*100, ndigits=DECIMALS))
        cv2.putText(image, txt, (x1+RECTANGLES_THICKNESS, max(0,y2-RECTANGLES_THICKNESS)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                    color=BGR_BLUE, thickness=LETTERS_SIZE)
        # Print Distance
        txt = 'Distance: {yDist} m'.format(yDist=round(dist_y/100, ndigits=DECIMALS))
        cv2.putText(image, txt, (x1+RECTANGLES_THICKNESS, max(0, y1 + (RECTANGLES_THICKNESS+LETTERS_SIZE)*4)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45, color=BGR_BLUE, thickness=LETTERS_SIZE)

        txt = 'To center: {xDist} m'.format(xDist=round(dist_x / 100, ndigits=DECIMALS))
        cv2.putText(image, txt,
                    (x1 + RECTANGLES_THICKNESS, max(0, y1 + (RECTANGLES_THICKNESS + LETTERS_SIZE) * 4*2)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45, color=BGR_BLUE, thickness=LETTERS_SIZE)

        # Print Name
        txt = name
        cv2.putText(image, txt, (x1 + RECTANGLES_THICKNESS, max(0, y1 - RECTANGLES_THICKNESS)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                    color=BGR_BLUE, thickness=LETTERS_SIZE)
    if swap_to_RGB:
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
    cv2.imshow("Detector",image)
    cv2.waitKey(20)
    return image

