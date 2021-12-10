
import pandas as pd
import pickle
import numpy as np
import cv2
import glob
import random
import numpy
import shutil


def doall():
    onlyfiles = glob.glob(r'/Users/botcha_mac/Courses/Vision/Testing_set2/image' + '/*.jpg', recursive=False)
    imgfiles = onlyfiles
    txtfiles = []
    file_list = []
    out_name = []
    a_file = open(r'/Users/botcha_mac/Courses/Vision/Testing_set2/testing_set.txt', 'r')
    lines = a_file.readlines()
    for line in lines:
        contents_split = line.split(' ')
        img_file, txt_file, out = contents_split
        img_file = 'testimgs/' + img_file[-15:]
        out = out[:-1]
        txtfiles.append(txt_file)
        file_list.append(img_file)
        out_name.append(out)
    # txtfiles = [w[10:] for w in contents_split if w.endswith('.txt')]

    # file_list = ['testimgs/' + im[-15:] for im in onlyfiles]

    humbi_joint_order = ['cnose', 'neck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee',
                         'rankle', 'lhip', 'lknee',
                         'lankle', 'reye', 'leye', 'rear', 'lear']

    vunet_order = ['cnose', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle',
                   'lhip', 'lknee',
                   'lankle', 'reye', 'leye']

    txtpath = r'/Users/botcha_mac/Courses/Vision/Testing_set2/'

    joints_norm = []
    for txt in txtfiles:
        data = pd.read_csv(txtpath + txt, sep=' ', header=None)
        data.set_index(pd.Index(humbi_joint_order), inplace=True)
        data_used = data[data.index.isin(vunet_order)]
        joints = data_used.to_numpy(dtype='float32')
        joints = np.divide(joints, 256)
        joints_norm.append(joints)

    # print(file_list)
    # print(joints_norm)
    test = [False for i in range(len(joints_norm))]
    indx = list(range(len(test)))
    print('indx', indx)
    num = int(len(test) * 0.5)
    print(len(test) - num)
    # print('index ', indx, "   len(train)-num", len(train))
    ran_indx = random.sample(indx, len(test) - num)
    # print(ran_indx)
    Test = numpy.array(test)
    Test[ran_indx] = False
    test1 = Test.tolist()
    # print(test1)
    p_file = {
        'joint_order': vunet_order,
        'imgs': file_list,
        'joints': joints_norm,
        'test': test1,
        'out_name': out_name
    }

    # save = r'D:\E\umn\CSCI\csci5561-python\hw3/index_1.p'
    save = r'/Users/botcha_mac/Courses/Vision/Testing_set2/test_index.p'

    with open(save, 'wb') as handle:
        pickle.dump(p_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    doall()
