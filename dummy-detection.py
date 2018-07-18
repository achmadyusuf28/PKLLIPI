import numpy as np
import cv2

gray = cv2.imread('tangga-3.jpg')
image_copy = gray.copy()
a, b, c = gray.shape
edges = cv2.Canny(gray,100,200,apertureSize = 3)
blank = np.zeros([a, b], dtype=np.uint8)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
# sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3) #16bit sign, untuk menyimpan nilai negatif
# sobelx = cv2.convertScaleAbs(sobelx) #fungsi untuk memutlakkan nilai negatif
# sobely = cv2.convertScaleAbs(sobely)
# edges_blur = (sobelx/2) + (sobely/2)


# retval, edge = cv2.threshold(edges_blur, 0, 255, cv2.THRESH_OTSU)


# invert = 255 - edge
# se = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, se)

minLineLength= 200
# minLineLength= 319
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength, maxLineGap=15)
print 'lines:', lines

# coba line

#
#

try:
    a,b,c = lines.shape
    for i in range(a):
        cv2.line(blank, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(image_copy, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 0), 3, cv2.LINE_AA)

    points = np.argwhere(blank == 255)  # find where the whit e pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
    crop = gray[y:y + h, x:x + w]  # create a cropped region for display
    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 0), 3)


    cv2.imshow('Detection', gray)
    cv2.imshow('Image copy', image_copy)
    cv2.imshow('Line only', blank)
    cv2.imshow('Crop', crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print 'No line detected.'

# cv2.imshow('Edge', edges)
# cv2.imshow('Opening', opening)
# cv2.waitKey(0)
# cv2.destroyAllWindows()