# Hermon Jay 14-10-2017
# klasifikasi jenis kelamin dengan 
# Decision Tree, SVM, KNN, dan Naive Bayes
import csv
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
def show(dt_awal, dt_pred):
    print('AWAL    |    PREDIKSI')
    for i in range(len((dt_pred))):
        print(dt_awal[i],' ',dt_pred[i])
# model untuk ketiga classifier
cDT = tree.DecisionTreeClassifier()
cSVM = svm.SVC()
cKNN = neighbors.KNeighborsClassifier()
cNB = GaussianNB()

# data latih
# [tingi, berat, ukuran_sepatu]
X, Y = read_data('datasetnocrop.csv')

# latih classifier
cDT = cDT.fit(X, Y)
cSVM = cSVM.fit(X, Y)
cKNN = cKNN.fit(X, Y)
cNB = cNB.fit(X, Y)

# data test
X_test, Y_test = read_data('beneran.csv')
# prediksi data test
Y_DT = cDT.predict(X_test)
Y_SVM = cSVM.predict(X_test)
Y_KNN = cKNN.predict(X_test)
Y_NB = cNB.predict(X_test)

# print prediksi
print("Prediksi Decision Tree : ", show(Y_test, Y_DT))
print("Prediksi SVM : ", show(Y_test, Y_SVM))
print("Prediksi KNN : ", show(Y_test, Y_KNN))
print("Prediksi Naive Bayes : ", show(Y_test, Y_NB))

# print akurasi
print("Akurasi Decision Tree : ", accuracy_score(Y_test, Y_DT))
print("Akurasi SVM : ", accuracy_score(Y_test, Y_SVM))
print("Akurasi KNN : ", accuracy_score(Y_test, Y_KNN))
print("Akurasi Naive Bayes : ", accuracy_score(Y_test, Y_NB))