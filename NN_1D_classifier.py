# this file is based on the codes linked below
# base : https://keras.io/getting_started/intro_to_keras_for_researchers/
# plus : https://www.linkedin.com/pulse/multi-task-supervised-unsupervised-learning-code-ibrahim-sobh-phd/


from keras.datasets import mnist
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
import keras as K
import keras
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import glob
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■ Classification Train ■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

df2 = pd.read_csv(r'.\data\train_freezed.csv')
err_train = df2.values[:, :-1]
status_train = df2.values[:, -1].astype(np.int64)

df3 = pd.read_csv(r'.\data\test_freezed.csv')
err_test = df3.values[:, :-1]
status_test = df3.values[:, -1].astype(np.int64)


x_train = err_train         # 이걸 쓰면 z출력을 tanh 로 활성화해야함.
y_train = status_train
y_train_cat = to_categorical(y_train)

x_test = err_test           # 이걸 쓰면 z출력을 tanh 로 활성화해야함.
y_test = status_test
y_test_cat = to_categorical(y_test)

# minMaxScaler = MinMaxScaler()
# print(minMaxScaler.fit(x_train))
# x_train = minMaxScaler.transform(x_train)
#
# minMaxScaler = MinMaxScaler()
# print(minMaxScaler.fit(x_test))
# x_test = minMaxScaler.transform(x_test)


#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■ this is for various condition ■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

var = [ 4e-5,8e-5, 12e-5]

for i in range(len(var)):

    var_str = 'lr replay %d th' % i
    dense1 = 16
    dense2 = 16
    train_epoch = 160
    batch_size = 300
    classes = 7
    learn_rate = var[i]
    #learn_rate = 4e-5
    original_dim = 10

    # ■■■■■■■■■■■■■■■■■■
    # ■■■■■■   NN   ■■■■■■■
    # ■■■■■■■■■■■■■■■■■■
    sc_input_img=Input(shape=(original_dim,), name='input')
    # Encoder: input to Z
    x = Dense(dense1, activation='relu', name='nn_1')(sc_input_img)
    x = Dense(dense2, activation='relu', name='nn_2')(x)

    predicted = Dense(classes, activation='softmax', name='class_output')(x)
    # Take input and give classification and reconstruction
    supervisedclassifier = Model(sc_input_img, predicted)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    supervisedclassifier.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['acc'])
    supervisedclassifier.summary()


    MODEL_SAVE_FOLDER_PATH = './model/replay %d th lr 0.00004 batch 300 dense 16/'  % i
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_acc:.4f}.hdf5'

    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    earlystopping = EarlyStopping(monitor='val_acc',  # 모니터 기준 설정 (val loss)
                                  patience=50,
                                  )
    # Single-Task Train
    SC_history = supervisedclassifier.fit(x_train, y_train_cat,
                                          epochs=train_epoch, batch_size=batch_size, shuffle=True, verbose=1,
                                          validation_data=(x_test, y_test_cat), callbacks=[checkpoint, earlystopping])

    plt.figure(2*i-1)
    plt.plot(SC_history.history['loss'], label='Train NN')
    plt.plot(SC_history.history['val_loss'], label='Test NN')
    plt.title('Model loss, %s =  %.5f '% (var_str , (var[i])))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    figname1 = 'Model loss %s =  %.5f.png'% (var_str , (var[i]))
    model_path1 = MODEL_SAVE_FOLDER_PATH + figname1
    plt.savefig(model_path1)

    plt.figure(2*i)
    test_pred = np.argmax(supervisedclassifier.predict(x_test), axis=1)
    print(test_pred)
    print(classification_report(y_test, test_pred))
    print("PRE: %.5f" % precision_score(y_test, test_pred, average='micro'))
    print("REC: %.5f" % recall_score(y_test, test_pred, average='micro'))
    print("F1: %.5f" % f1_score(y_test, test_pred, average='micro'))

    f = open('./results/%s _results.txt' % var_str, 'a')  # 파일 열기
    temp = "%s =  %.5f   " % (var_str , (var[i]))
    f.write(temp)
    temp = "PRE: %.5f \n"  % precision_score(y_test, test_pred, average='micro')
    f.write(temp)
    f.close()

    confmat = confusion_matrix(y_true=y_test, y_pred=test_pred)
    sns.heatmap(confmat, annot = True, fmt = 'd', cmap = 'Blues')
    plt.title('supervised classifier, %s = %.5f ' % (var_str , (var[i])))
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    figname2 = 'SC Confusion Matrix, %s =  %.5f.png' % (var_str , (var[i]))
    model_path2 = MODEL_SAVE_FOLDER_PATH + figname2
    plt.savefig(model_path2)


