from Constants import *
import matplotlib as mtb
try:
    mtb.use('GTK3Cairo')
except:
    from warnings import warn
    warn("PyCairo is not in use. If you are not viewing any plot install pycairo")
import cv2

def show_detections(image, boxes, names, distances, confidences):
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

    #cv2.imshow("Detector",image)
    #cv2.waitKey(20)
    mtb.pyplot.figure("RobotEye")
    mtb.pyplot.clf()
    mtb.pyplot.imshow(cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB))
    mtb.pyplot.pause(.000001)
    return image

