import cv2 as cv
import os
import numpy as np

def find_circles(path, source):
    size = len(os.listdir(path))
    for i in range(size):
        pred_path = path + "pred_{}.jpg".format(i)
        source_path = source + "input.{}.jpg".format(i)
        img = cv.imread(pred_path, 1)
        output =  cv.imread(source_path, 1)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,3, 10)
        results = []
        if circles is not None:
            print("circle found")
            circles = np.round(circles[0, :]).astype("int")
            (x,y,r) = circles[0]
            cv.circle(output, (x, y), r, (0, 255, 0), 4)
            results.append((i, circles[0]))
            # cv.imwrite(filepath, output)
        else:
            print("Circle not found")
            results.append(None)

if __name__ == "__main__":
    path = "./run1/"
    find_circles(path)
