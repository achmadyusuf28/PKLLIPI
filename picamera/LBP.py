import cv2
import numpy as np
import csv

def ambil_pixel(gambar,center,x,y):
    new_value=0
    try:
        if (x == -1 or y == -1):
            new_value = 0
        elif gambar[x][y]>=center:
            new_value = 1
        else:
            new_value=0
    except:
        pass
    return new_value

def lbp_histogram(hist):
    histogram=np.zeros(len(hist))
    for i in range(len(hist)):
        histogram[i]=hist[i]
    return histogram

def save_data(data):
    with open('test.csv'):
        return

def hitung_nilai_lbp(gambar,x,y):
    center=gambar[x][y]
    value=[]
    value.append(ambil_pixel(gambar, center, x-1, y-1))
    value.append(ambil_pixel(gambar, center, x-1, y))
    value.append(ambil_pixel(gambar, center, x-1, y+1))
    value.append(ambil_pixel(gambar, center, x, y+1))
    value.append(ambil_pixel(gambar, center, x+1, y+1))
    value.append(ambil_pixel(gambar, center, x+1, y))
    value.append(ambil_pixel(gambar, center, x+1, y-1))
    value.append(ambil_pixel(gambar, center, x, y-1))

    nilai_pangkat=[1,2,4,8,16,32,64,128]
    nilai_lbp=0
    for i in range(len(nilai_pangkat)):
        if(value[i]==0):
            nilai_lbp=nilai_lbp+0
        else:
            nilai_lbp=nilai_lbp+nilai_pangkat[i]

    return nilai_lbp

def main(gambar):
    img=cv2.imread(gambar);
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    heigh, width = gray.shape
    hasil=np.zeros((heigh, width),np.uint8)
    for x in range(len(gray)):
        for y in range(len(gray[x])):
            hasil[x, y]=hitung_nilai_lbp(gray, x, y)
    hist_lbp = cv2.calcHist([hasil], [0], None, [256], [0, 256])
    hist_jadi_data= lbp_histogram(hist_lbp)
    masuk_histogram=[i for i in hist_jadi_data]

    cv2.imshow('test',hasil)
    cv2.imshow('1',img)
    cv2.waitKey(0)
    print(hasil)
    cv2.destroyAllWindows()
    return masuk_histogram

# if __name__ == '__main__':
#     test=[]
#     test.append(main("1.jpg"))
#     test.append(main("2.jpg"))
#     test.append(main("3.jpg"))
#     print test
if __name__ == '__main__':
    main("1.jpg")