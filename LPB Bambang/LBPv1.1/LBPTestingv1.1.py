import cv2, csv
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

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

def show(dt_awal, dt_pred):
    count = 0
    print('             AWAL    | PREDIKSI')
    for i in range(len((dt_pred))):
        if dt_awal[i] != dt_pred[i]:
            count += 1
        print('Data ke-', i+1, '. ', dt_awal[i],'  ',dt_pred[i])
    return count
        
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
    return n_bin

if __name__ == '__main__':
    datauji = []
    kelasuji = []
    
    # data latih
    data, kelas = read_data('datasetNoBinTraining.csv')
    
    for i in range(1, 39):
        print('Testing data ke-',i)
        img = '../datasets/testing_crop/2/ (' + str(i) +').jpg'
        datauji.append(main(img,'female'))
        if i < 22:
            kelasuji.append('male')
        else:
            kelasuji.append('female')
        
    cDT = tree.DecisionTreeClassifier()
    cSVM = svm.SVC()
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
    print("Kelasalahan Decision Tree : ", show(kelasuji, Y_DT), '\n')
    print("Kelasalahan SVM : ", show(kelasuji, Y_SVM), '\n')
    print("Kelasalahan KNN : ", show(kelasuji, Y_KNN), '\n')
    print("Kelasalahan Naive Bayes : ", show(kelasuji, Y_NB), '\n')
    
    # print akurasi
    print("Akurasi Decision Tree : ", accuracy_score(kelasuji, Y_DT)*100, '%')
    print("Akurasi SVM : ", accuracy_score(kelasuji, Y_SVM)*100, '%')
    print("Akurasi KNN : ", accuracy_score(kelasuji, Y_KNN)*100, '%')
    print("Akurasi Naive Bayes : ", accuracy_score(kelasuji, Y_NB)*100, '%')
    print("LBP Testing is finished")