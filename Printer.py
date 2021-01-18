from Constants import *
import cv2

def show_detections(image, boxes, names=None, distances = None, confidences = None):
    """
    Plots in the screen an image with the detections drawn in it (updates when called again).
    :param image: Numpy. Image to plot in BGR format.
    :param boxes: List of rectangles in format (x1, y1, x2, y2). Boxes where the detections are.
    :param names: List of String. List of the names associated to each location.
    :param distances: List of Float. List of the distances associated to each location.
    :param confidences: List of Float. List of confidences associated to each location
    :return: Numpy. Image with all the detections and associated information drawn in it.
    """
    if names is None: names = [None for _ in boxes]
    if confidences is None: confidences = [None for _ in boxes]
    if distances is None: distances = [(None, None) for _ in boxes]
    for (x1, y1, x2, y2), name, (dist_x, dist_y), confidence in zip(boxes, names, distances, confidences):
        # Print confidence
        cv2.rectangle(image, (x1, y1), (x2, y2), color=BGR_BLUE, thickness=RECTANGLES_THICKNESS)
        if confidence is not None:
            txt = 'Conf: {conf}%'.format(conf=round(confidence*100, ndigits=DECIMALS))
            cv2.putText(image, txt, (x1+RECTANGLES_THICKNESS, max(0,y2-RECTANGLES_THICKNESS)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                        color=BGR_BLUE, thickness=LETTERS_SIZE)
        if dist_y is not None:
            # Print Distance
            txt = 'Distance: {yDist} m'.format(yDist=round(dist_y/100, ndigits=DECIMALS))
            cv2.putText(image, txt, (x1+RECTANGLES_THICKNESS, max(0, y1 + (RECTANGLES_THICKNESS+LETTERS_SIZE)*4)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.45, color=BGR_BLUE, thickness=LETTERS_SIZE)
        if dist_x is not None:
            txt = 'To center: {xDist} m'.format(xDist=round(dist_x / 100, ndigits=DECIMALS))
            cv2.putText(image, txt,
                        (x1 + RECTANGLES_THICKNESS, max(0, y1 + (RECTANGLES_THICKNESS + LETTERS_SIZE) * 4*2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.45, color=BGR_BLUE, thickness=LETTERS_SIZE)
        if name is not None:
            # Print Name
            txt = name
            cv2.putText(image, txt, (x1 + RECTANGLES_THICKNESS, max(0, y1 - RECTANGLES_THICKNESS)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                        color=BGR_BLUE, thickness=LETTERS_SIZE)

    cv2.imshow("Detector",image)
    cv2.waitKey(1)
    return image

