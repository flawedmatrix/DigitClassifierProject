import matplotlib.pyplot as plt
import csv
import IPython
import numpy as np

if __name__ == '__main__':
    f_slms_te = open('nntrainer/slms_test.txt', 'r')
    f_slms_tr = open('nntrainer/slms_train.txt', 'r')
    f_slce_te = open('nntrainer/slce_test.txt', 'r')
    f_slce_tr = open('nntrainer/slce_train.txt', 'r')
    f_mlms_te = open('nntrainer/mlms_test.txt', 'r')
    f_mlms_tr = open('nntrainer/mlms_train.txt', 'r')
    f_mlce_te = open('nntrainer/mlce_test.txt', 'r')
    f_mlce_tr = open('nntrainer/mlce_train.txt', 'r')

    csvr = csv.reader(f_slms_te);
    slms_te = list(csvr)[0]
    slms_te = [np.log(float(v)*100) for v in slms_te]

    csvr = csv.reader(f_slms_tr);
    slms_tr = list(csvr)[0]
    slms_tr = [np.log(float(v)*100) for v in slms_tr]

    csvr = csv.reader(f_slce_te);
    slce_te = list(csvr)[0]
    slce_te = [np.log(float(v)*100) for v in slce_te]

    csvr = csv.reader(f_slce_tr);
    slce_tr = list(csvr)[0]
    slce_tr = [np.log(float(v)*100) for v in slce_tr]

    csvr = csv.reader(f_mlms_te);
    mlms_te = list(csvr)[0]
    mlms_te = [np.log(float(v)*100) for v in mlms_te]

    csvr = csv.reader(f_mlms_tr);
    mlms_tr = list(csvr)[0]
    mlms_tr = [np.log(float(v)*100) for v in mlms_tr]

    csvr = csv.reader(f_mlce_te);
    mlce_te = list(csvr)[0]
    mlce_te = [np.log(float(v)*100) for v in mlce_te]

    csvr = csv.reader(f_mlce_tr);
    mlce_tr = list(csvr)[0]
    mlce_tr = [np.log(float(v)*100) for v in mlce_tr]

    # Epoch from 0 to 200
    e = range(201)
    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.plot(e, slms_tr, '-', e, slms_te, '-')
    sp.set_title("Single Layer Neural Network with Mean Squares")
    sp.set_xlabel("Epoch")
    sp.set_ylabel("log(%error)")
    plt.savefig('slms.png', format='png')

    fig = plt.figure()
    sp = fig.add_subplot(111)
    plt.plot(e, slce_tr, '-', e, slce_te, '-')
    sp.set_title("Single Layer Neural Network with Cross Entropy")
    sp.set_xlabel("Epoch")
    sp.set_ylabel("log(%error)")
    plt.savefig('slce.png', format='png')

    fig = plt.figure()
    sp = fig.add_subplot(111)
    plt.plot(e, mlms_tr, '-', e, mlms_te, '-')
    sp.set_title("Multi Layer Neural Network with Mean Squares")
    sp.set_xlabel("Epoch")
    sp.set_ylabel("log(%error)")
    plt.savefig('mlms.png', format='png')

    fig = plt.figure()
    sp = fig.add_subplot(111)
    plt.plot(e, mlce_tr, '-', e, mlce_te, '-')
    sp.set_title("Multi Layer Neural Network with Cross Entropy")
    sp.set_xlabel("Epoch")
    sp.set_ylabel("log(%error)")
    plt.savefig('mlce.png', format='png')