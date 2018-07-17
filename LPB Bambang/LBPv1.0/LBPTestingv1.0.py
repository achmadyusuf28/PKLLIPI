import cv2, csv
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# membaca data training
def read_data(loc):
    dataTraining=[]
    kelas=[]
    with open(loc) as f:
        readCSV = csv.reader(f, delimiter=',')
        for row in readCSV:
            data=[]
            for i in range(len(row)):
                if i != len(row)-1:
                    data.append(float(row[i]))
                else:
                    kelas.append(row[i])
            dataTraining.append(data)
    return (dataTraining, kelas)

# mendapatkan nilai piksel citra          
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

# proses bin pada histogram menjadi 6 bin setiap bin rentang 32
def lbp_in(hist):
    his=np.zeros((6))
    for i in range(len(hist)):
        if i<32:
            his[0]= his[0]+hist[i]
        elif i<64:
            his[1]= his[1]+hist[i]
        elif i<96:
            his[2]= his[2]+hist[i]
        elif i<128:
            his[3]= his[3]+hist[i]
        elif i<192:
            his[4]= his[4]+hist[i]
        else:
            his[5]= his[5]+hist[i]
    return his

# perhitungan algoritma LBP
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

# menghitung nilai kesalahan prediksi
def error(dt_awal, dt_pred):
    count = 0
    for i in range(len((dt_pred))):
        if dt_awal[i] != dt_pred[i]:
            count += 1
    return count
    
def main(image_file, kelas, num):
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
    if num == '1':
        cv2.imshow(num, img_lbp)
    return n_bin

if __name__ == '__main__':
    datauji = []
    kelasuji = []
    
    # data latih
    data, kelas = read_data('datasetBinTraining.csv')
    
    #proses testing 20 data
    for i in range(1, 40):
        print('Testing data ke-',i)
        img = '../datasets/testing_crop/ganda/ (' + str(i) +').jpg'
        datauji.append(main(img,'female', str(i)))
        if i < 22:
            kelasuji.append('female')
        else:
            kelasuji.append('male')
    print()
    gmb = cv2.imread('gambar.png')
    cv2.imshow('Data uji', gmb)
    # create object    
    cDT = tree.DecisionTreeClassifier()
    cSVM = svm.SVC(kernel='rbf', C=1000, gamma=1)
    cKNN = neighbors.KNeighborsClassifier()
    cNB = GaussianNB()
    
    # Data latih classifier
    cDT = cDT.fit(data, kelas)
    cSVM = cSVM.fit(data, kelas)
    cKNN = cKNN.fit(data, kelas)
    cNB = cNB.fit(data, kelas)
    
    # prediksi data test
    Y_DT = cDT.predict(datauji)
    Y_SVM = cSVM.predict(datauji)
    Y_KNN = cKNN.predict(datauji)
    Y_NB = cNB.predict(datauji)
    
    # print prediksi
    print('-------------------------------------------------------------------------------')    
    print('  Data Asli   | Decision Tree |      SVM      |      K-NN     |  Naive Bayes  |')
    print('-------------------------------------------------------------------------------')
    for i in range(0,39):
        print('%-14s|%-15s|%-15s|%-15s|%-15s|' % (kelasuji[i], Y_DT[i], Y_SVM[i], Y_KNN[i], Y_NB[i]))
    print('-------------------------------------------------------------------------------')
    print('%-14s|   %-12s|   %-12s|   %-12s|   %-12s|' % ('Jumlah Error', error(kelasuji, Y_DT), error(kelasuji, Y_SVM), error(kelasuji, Y_KNN), error(kelasuji, Y_NB)))
    print('-------------------------------------------------------------------------------')
    
    # print akurasi
    print()
    print('%-25s %-1s %-4.2f %-1s' % ('Akurasi Decision Tree',':', accuracy_score(kelasuji, Y_DT)*100, '%'))
    print('%-25s %-1s %-4.2f %-1s' % ('Akurasi SVM',':', accuracy_score(kelasuji, Y_SVM)*100, '%'))
    print('%-25s %-1s %-4.2f %-1s' % ('Akurasi K-NN',':', accuracy_score(kelasuji, Y_KNN)*100, '%'))
    print('%-25s %-1s %-4.2f %-1s' % ('Akurasi Naive Bayes',':', accuracy_score(kelasuji, Y_NB)*100, '%'))
    print("LBP Testing is finished")
    cv2.waitKey(0)
    cv2.destroyAllWindows()