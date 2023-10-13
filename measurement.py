import sys
import serial
import numpy as np
import time
import pyautogui
import os
import datetime


def createpath(path):
    if not os.path.exists(path):
        os.mkdir(path)


def isint(s):  # 整数値を表しているかどうかを判定
    try:
        int(s, 10)  # 文字列を実際にint関数で変換してみる
    except ValueError:
        return False  # 例外が発生＝変換できないのでFalseを返す
    else:
        return True  # 変換できたのでTrueを返す


def is_num_delimiter(s):
    try:
        float(s.replace(',', ''))
    except ValueError:
        return False
    else:
        return True


def recieve_ain_values(ser):
    line = ser.readline()
    line_disp = line.strip().decode('UTF-8', errors='ignore')
    line_elements = line_disp.split(",")

    if is_num_delimiter(line_disp) == False:
        return 0

    if len(line_elements) != num_sensor:
        return 0

    ain_values = []
    for i in range(num_sensor):
        if isint(line_elements[i]):
            # '''
            if 100 < int(line_elements[0]) < 1024:
                ain_values.append(int(line_elements[i]))
            else:
                return 0
                # ain_values.append(1)
            # '''
            # ain_values.append(int(line_elements[i]))
        else:
            return 0

    return ain_values


def loop():
    try:
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = 'COM3'
        ser.dsrdtr = True
        ser.open()
    except:
        print("No Arduino found", file=sys.stderr)
        raise

    ser.write(str.encode('1\n\r'))


def Timer(secs):
    for s in range(secs, -1, -1):
        print(s)
        time.sleep(1)
    print("start")


if __name__ == "__main__":
    # settings
    sup = 10.5
    exh = 16.5
    late = 0.005
    count = int((sup + exh) / late)
    j = 0
    num_sensor = 4
    object_name = ['air', 'bottle', 'can', 'pet',  'pla']

    now = datetime.datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M")
    dir_for_learning = "./learning/" + current_time
    createpath(dir_for_learning)

    try:
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = 'COM4'
        ser.dsrdtr = True
        ser.timeout = 0
        ser.open()
    except:
        print("No Arduino found", file=sys.stderr)
        raise

    try:
        print("Press Ctrl+C to quit")

        for object in object_name:
            ser.reset_input_buffer()
            filepath = f'{dir_for_learning}\\{object}.csv'
            # measurement
            i = 0
            value = np.zeros([count, num_sensor])
            Timer(5)
            # initial data
            initial = recieve_ain_values(ser)
            while initial == 0:
                initial = recieve_ain_values(ser)
            print(initial)
            prev = time.time()
            ser.reset_input_buffer()
            loop()
            while i < count:
                current = time.time()
                if((current - prev) >= late):
                    prev += late
                    ain_values = recieve_ain_values(
                        ser)
                    if ain_values != 0:
                        for k in range(4):
                            ain_values[k] -= initial[k]
                        value[i] = ain_values
                        print(value[i])
                        i += 1
            np.savetxt(filepath, value, delimiter=',')
            j += 1

    except KeyboardInterrupt:
        ser.close()
