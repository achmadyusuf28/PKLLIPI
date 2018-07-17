import cv2
import numpy as np
import csv

def save_data(loc, a):
    with open(loc, 'w') as cvsfile:
        spamwriter = csv.writer(cvsfile, lineterminator='\n')
        for i in a:
           spamwriter.writerow(i)

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_in(hist):
    his=np.zeros((len(hist)))
    for i in range(len(hist)):
        his[i]= hist[i]
    return his

def lbp_calculated_pixel(img, x, y):
    '''
     1  |   2 |   4
    ----------------
    128 |   0 |   8
    ----------------
     64 |  32 |   16
    '''
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left


    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def main(image_file, kelas):
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    lbp_bin = lbp_in(hist_lbp)
    n_bin = [i for i in lbp_bin]
    n_bin.append(kelas)
    return n_bin

if __name__ == '__main__':
    a=[]
    for i in range(1,286): #female 285
        print('Training data female ke-',i)
        img = '../datasets/training_crop/1female/female (' + str(i) +').jpg'
        a.append(main(img,'female'))
    for i in range(1,253): #male 252
        print('Training data male ke-',i)
        img = '../datasets/training_crop/2male/male ('+ str(i) +').jpg'
        a.append(main(img,'male'))
    save_data('datasetNoBinTraining.csv', a)
    print("LBP Training is finished")