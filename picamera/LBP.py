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

def lbp_histogram1(hist):
    histogram=np.zeros(len(hist))
    for i in range(len(hist)):
        histogram[i]=hist[i]
    return histogram

def lbp_histogram2(hist):
    histogram=np.zeros(6)

def hitung_nilai_lbp(gambar,x,y):
    '''
         1  |   2 |   4
        ----------------
        128 |   0 |   8
        ----------------
         64 |  32 |   16
        '''
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

def main(gambar,kelas):
    img=cv2.imread(gambar);
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lebar, panjang = gray.shape

#testing input
# img=cv2.imread('anone.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# lebar,panjang=gray.shape
# print gray
# print gray[-1][-1]

#test ilung lbp2
img=cv2.imread('anone.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lebar,panjang=gray.shape
print lebar
print panjang
hasil=[]
for x in range(len(gray)):
    for y in range(len(gray[x])):
        hasil.append(hitung_nilai_lbp(gray,x,y))
tulis=open("testing.txt","r+")
tulis.write(hasil)
tulis.close()