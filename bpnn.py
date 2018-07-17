import csv
import math
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
    return sumproduct(a, b) + bias

def sumproduct(a, b):
    counter = 0
    for i in range(len(a)):
        counter += a[i]*b[i]
    return counter

def func_activation(a):
    return 1 / (1 + math.exp(-(a)))

def diff_activation(a):
    return a * (1-a)

def error(t, y):
    sum = 0
    for i in range(len(t)):
        sum += pow((float(t[i])-y[i]), 2)
    return sum/len(t)

#
##
# pembatas fungsi
##
#

data, t = read_data('xor.csv')
hid_layer = 2
alpha = 1
err = 1.0
min_err = 0.1
# --> err > min_err
max_epoch = 39
# --> epoch < max_epoch
epoch = 0
# weight = np.random.uniform(-1, 1, size=[len(data[0]), hid_layer])
# bias = np.random.uniform(-1, 1, size=hid_layer)
# v = np.random.uniform(-1, 1, size=hid_layer+1)

# feedfoward
weight = np.array([[0.1, -0.5], [0.2, -0.2]], dtype=float)
bias = np.array([0.3, 0.1], dtype=float)
v = np.array([-0.1, -0.3, 0.4], dtype=float)

while err > min_err:
    print 'Epoch ke-', epoch+1
    for k in range(len(data)):
        if k == 0:
            z_in = np.zeros((len(data[k])), dtype=float)
            z = np.zeros((len(z_in)), dtype=float)
            d_v = np.zeros((len(z) + 1), dtype=float)
            d_in = np.zeros((len(v) - 1), dtype=float)
            d_z = np.zeros((len(z)), dtype=float)
            d_w = np.zeros(([len(weight), len(weight[k])]), dtype=float)
            d_b = np.zeros((len(bias)), dtype=float)
            y_epoch = np.zeros((len(t)), dtype=float)

        for i in range(len(z_in)):
            z_in[i] = calculate_zin(data[k], weight[i], bias[i])

        for i in range(len(z)):
            z[i] = func_activation(z_in[i])

        y_in = sumproduct(z, v) + v[len(v)-1]

        y = func_activation(y_in)
        y_epoch[k] = y

        # backward
        d_y = (float(t[k]) - y) * diff_activation(y)

        for i in range(len(d_v)):
            j = i
            if i > len(z)-1:
                d_v[i] = alpha * d_y
            else:
                d_v[i] = alpha*d_y*z[i]

        for i in range(len(d_in)):
            d_in[i] = d_y * v[i]

        for i in range(len(d_z)):
            d_z[i] = diff_activation(z[i]) * d_in[i]

        for i in range(len(d_w)):
            for j in range(len(d_w[i])):
                x = data[k][j]
                y = d_z[i]
                d_w[i][j] = alpha * data[k][j] * d_z[i]

        for i in range(len(d_b)):
            d_b[i] = alpha * d_z[i]

        # update bobot
        for i in range(len(weight)):
            for j in range(len(weight[i])):
                weight[i][j] = weight[i][j] + d_w[i][j]

        for i in range(len(bias)):
            bias[i] = bias[i] + d_b[i]

        for i in range(len(v)):
            v[i] = v[i] + d_v[i]

        print 'Data ke-', k+1
        print 'w:', weight
        print 'b:', bias
        print 'v:', v
        print

    # hitung error
    err = error(t, y_epoch)
    print 'err:', err, '\n'
    epoch += 1
