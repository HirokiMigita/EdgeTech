import collections as col
import glob as gl
import time

import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)

from model_gpu import ESN, Tikhonov

cp.random.seed(seed=0)


# 出力のスケーリング


class ScalingShift:
    def __init__(self, scale, shift):
        '''
        :param scale: 出力層のスケーリング（scale[n]が第n成分のスケーリング）
        :param shift: 出力層のシフト（shift[n]が第n成分のシフト）
        '''
        self.scale = cp.diag(scale)
        self.shift = cp.array(shift)
        self.inv_scale = cp.linalg.inv(self.scale)
        self.inv_shift = -cp.dot(self.inv_scale, self.shift)

    def __call__(self, x):
        return cp.dot(self.scale, x) + self.shift

    def inverse(self, x):
        return cp.dot(self.inv_scale, x) + self.inv_shift


def teacher_label(data):
    length = len(data)
    label = cp.zeros([length, num_object])

    for i in range(num_object):
        j = 0
        for j in range(int(length/num_object)):
            #label[j+length/num_object*i][i] = 1
            if j < before_touch:
                label[j+length/num_object*i][0] = 1
            if before_touch <= j < after_touch:
                label[j+length/num_object*i][i] = 1
            if after_touch <= j < int(length/num_object):
                label[j+length/num_object*i][0] = 1

    return label


def counter(data):
    length = len(data)/num_object
    matrix = cp.zeros([num_object, length])
    for i in range(num_object):
        for j in range(length):
            c = col.Counter(data[j])
            matrix[i][j] = c[i]

    return matrix


if __name__ == '__main__':
    num_object = 5
    num_sensor = 4
    before_touch = 500
    after_touch = 3800

    start_time = time.time()
    '''4object
    train_date = "2023_10_12_12_21"
    test_date = "2023_10_12_12_24"#almost same
    test_date = "2023_10_12_12_38"#right
    test_date = "2023_10_12_12_54"#left
    '''

    #train_date = "2023_10_12_19_00"
    train_date = "2023_10_13_11_33"
    test_date = "2023_10_13_11_36"  # almost same

    file_train = gl.glob(f"learning/{train_date}/*")
    file_train = natsorted(file_train)
    file_test = gl.glob(f"learning/{test_date}/*")
    file_test = natsorted(file_test)

    train_data = cp.empty([0, num_sensor])
    for filename in file_train:
        data = cp.loadtxt(filename, delimiter=',', dtype='float64')
        train_data = cp.vstack([train_data, data])

    test_data = cp.empty([0, num_sensor])
    for filename in file_test:
        data = cp.loadtxt(filename, delimiter=',', dtype='float64')
        test_data = cp.vstack([test_data, data])

    train_data = train_data.reshape(-1, num_sensor)*0.0025
    test_data = test_data.reshape(-1, num_sensor)*0.0025

    train_teacher = teacher_label(train_data)
    test_teacher = teacher_label(test_data)

    u = cp.concatenate([train_data, test_data])
    d = cp.concatenate([train_teacher, test_teacher])

    # 時系列入力データ生成
    period = 10
    T = len(train_data)
    t_object = int(len(test_data)/num_object)

    # 訓練・検証用情報
    train_U = u[:T].reshape(-1, num_sensor)
    train_D = d[:T]

    test_U = u[T:].reshape(-1, num_sensor)
    test_D = d[T:]

    # 出力のスケーリング関数
    output_func = ScalingShift(
        cp.full(num_object, 0.5), cp.full(num_object, 0.5))

    # ESNモデル
    N_x = 500  # リザバーのノード数
    alpha = 0.01
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1,
                input_scale=1, rho=0.9, leaking_rate=alpha,
                output_func=output_func, inv_output_func=output_func.inverse,
                classification=True, average_window=period)

    # 学習（リッジ回帰）
    train_Y = model.train(train_U, train_D,
                          Tikhonov(N_x, train_D.shape[1], 0.1))

    # 訓練データに対するモデル出力
    test_Y = model.predict(test_U)

    learning_time = time.time() - start_time
    print("learning_time:{0}".format(learning_time) + "[sec]")

    # 入力波形
    plt.rcParams['font.family'] = "MS Gothic"
    plt.figure(figsize=(6, 6))
    plt.xlim(0, len(u))
    u = cp.asnumpy(u)
    plt.plot(u, linewidth=2, linestyle='solid')
    plt.savefig(f"fig/Input_Waveform__{train_date}_{test_date}.svg", bbox_inches='tight')

    # 出力
    plt.figure(figsize=(16, 8))
    plt.xlim(0, len(u))
    output = cp.concatenate([train_Y, test_Y])
    output = cp.asnumpy(output)
    plt.plot(output, linewidth=2, linestyle='solid', label='Model')
    plt.gca().set_prop_cycle(None)
    d = cp.asnumpy(d)
    plt.plot(d, linewidth=2, linestyle='--', label='target')
    plt.xlabel('タイムステップ')
    plt.ylabel('出力')
    plt.axvline(x=T, color='k', linestyle=':')
    plt.legend(loc="best")
    plt.savefig(
        f"fig/Output_Results_{train_date}_{test_date}_{N_x}_{alpha}.svg", bbox_inches='tight')

    # 評価
    y_true = cp.argmax(test_teacher, axis=1)
    y_true = cp.asnumpy(y_true)
    true_label = []
    for i in range(num_object):
        c_true_before = col.Counter(y_true[i*t_object:i*t_object+before_touch])
        true_label.append(c_true_before.most_common()[0][0])
        c_true_touch = col.Counter(y_true[i*t_object+before_touch:i*t_object+after_touch])
        true_label.append(c_true_touch.most_common()[0][0])
        c_true_after = col.Counter(y_true[i*t_object+after_touch:i*t_object+t_object])
        true_label.append(c_true_after.most_common()[0][0])

    y_predict = cp.argmax(test_Y, axis=1)
    y_predict = cp.asnumpy(y_predict)
    predict_label = []
    for i in range(num_object):
        c_predict_before = col.Counter(y_predict[i*t_object:i*t_object+before_touch])
        predict_label.append(c_predict_before.most_common()[0][0])
        c_predict_touch = col.Counter(y_predict[i*t_object+before_touch:i*t_object+after_touch])
        predict_label.append(c_predict_touch.most_common()[0][0])
        c_predict_after = col.Counter(y_predict[i*t_object+after_touch:i*t_object+t_object])
        predict_label.append(c_predict_after.most_common()[0][0])

    label = ['air', 'bottle', 'can', 'pet',  'pla']
    cm_test = confusion_matrix(true_label, predict_label)
    cmd = ConfusionMatrixDisplay([cm_test, label])
    '''
    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    cmd.plot(cmap='Blues')
    ax.set_xticklabels(label)
    fig.autofmt_xdate(rotation=45)
    plt.title('Accuracy: {:.3f}'.format(score))
    plt.tight_layout()
    plt.savefig(
        f"integrate_result_{train_date}_{N_x}.svg", bbox_inches='tight')
    '''
    score = accuracy_score(true_label, predict_label)
    print(score)

    score *= 100
    plt.figure(figsize=(12, 10))
    plt.rcParams["font.size"] = 20
    ax = sns.heatmap(cm_test, annot=True, square=True, cmap='Blues',
                     xticklabels=label, yticklabels=label, linewidths=.5, linecolor='black')
    ax.set_xticklabels(label, rotation=45, ha='right')
    ax.set_title('正解率: {:.1f}%'.format(score))
    ax.set_ylabel('正解ラベル')
    ax.set_xlabel('予測ラベル')
    plt.tight_layout()
    plt.savefig(f"fig/heatmap_{train_date}_{test_date}_{N_x}.svg", bbox_inches='tight')

    plot_time = time.time() - start_time
    print("plot_time:{0}".format(plot_time) + "[sec]")

    plt.show()
