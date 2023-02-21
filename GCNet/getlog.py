import os
import re
import numpy as np
import pandas as pd

with open('./logs/2022-11-22-2233-50.log', 'r') as fr:
    logs = fr.readlines()
    train_log = []
    test_log = []
    for line in logs:
        info = re.findall(r'-?\d+\.?\d*', line)
        if 'test' in line:
            test_log.append(list(map(float,info[5:])))
        else:
            train_log.append(list(map(float,info[5:])))
    train_log = np.array(train_log)
    train_log = np.delete(train_log, -2, axis=1)
    test_log = np.array(test_log)
    test_log = np.delete(test_log, -2, axis=1)
    temp=test_log[:, 1].copy()
    test_log[:, 2] = np.sqrt(temp)

    # temp = train_log[:, -1].copy()
    # train_log[:, -1] = test_log[:, -1]
    # test_log[:, -1] = temp
    train_log = pd.DataFrame(train_log)
    train_log.columns = ['mae', 'mse', 'rmse', 'mape', 'r2']
    train_log.to_csv(os.path.join('train_log.csv'), index=False)

    test_log = pd.DataFrame(test_log)
    test_log.columns = ['mae', 'mse', 'rmse', 'mape', 'r2']
    test_log.to_csv(os.path.join('test_log.csv'), index=False)
