import csv
import numpy as np

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

def calculate_zin(a, b, bias):
    x = sumproduct(a, b)
    y = round(x+bias, 5)
    return y

def sumproduct(a, b):
    counter = 0
    for i in range(len(a)):
        x = a[i]
        y = b[i]
        counter += a[i]*b[i]
    return counter

data, kelas = read_data('xor.csv')
hid_layer = 2
# weight = np.random.uniform(-1, 1, size=[len(data[0]), hid_layer])
# bias = np.random.uniform(-1, 1, size=hid_layer)
# v = np.random.uniform(-1, 1, size=hid_layer+1)

# test (sama spt manualisasi)
weight = np.array([[0.1, -0.5], [0.2, -0.2]], dtype=float)
bias = np.array([0.3, 0.1], dtype=float)
v = np.array([-0.1, 0.3, 0.4], dtype=float)

print('zin[1]:', calculate_zin(data[0], weight[0], bias[0]))
print('zin[1]:', calculate_zin(data[0], weight[1], bias[1]))

z_in = np.zeros((len(data)), dtype=float)

for i in range(len(data)):
    for j in range(len(z_in)):
        calculate_zin(data[i], weight[j], bias[j])
print('z_in:', z_in)