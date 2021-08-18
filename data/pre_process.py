import numpy as np
import cv2
import os
from numpy.core.defchararray import center
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import glob
import shutil

from numpy.lib.function_base import diff

def pre_process(data_path="./images"):
    # input: 3*512*512
    # output: heatmap 512*512
    video_list = os.listdir(data_path)
    # images = []
    # gts = []
    count = 0
    for video in video_list:
        print(video)
        image_path = os.path.join(data_path, video, "image")
        gt_path = os.path.join(data_path, video, "gt")
        gt_file = open(os.path.join(gt_path, "gt.txt"), "r",encoding='utf-8')
        gt = []
        line = gt_file.readline() # 读取第一行
        while line:
            txt_data = eval(line) # 可将字符串变为元组
            gt.append(txt_data) # 列表增加
            line = gt_file.readline()
        gt_file.close()
        # print(gt)
        image_file_list = os.listdir(image_path)
        image_list = []
        for i in range(len(image_file_list)):
            image_gray = np.mean(mpimg.imread(os.path.join(image_path, image_file_list[i])), axis=2)
            image_list.append(image_gray)
        # background_diff_image_list = []
        # diff_img_list = []
        # final_img_list = []
        for i in range(10, len(image_list)):
            print(i)
            background_image = np.zeros_like(image_list[0])
            for j in range(10):
                background_image += np.uint8(image_list[i-j-1]/10)
            background_diff_image = signal.medfilt2d(np.clip(image_list[i]-background_image, 0, 255), (3,3))

            diff_img = signal.medfilt2d(np.clip(image_list[i]-image_list[i-1], 0, 255), (3,3))

            final_list = []
            final_list.append(image_list[i])
            final_list.append(background_diff_image)
            final_list.append(diff_img)

            final_img = np.array(final_list)
            # print(final_img.shape)
            # final_img_list.append(final_img)
            final_img = np.transpose(final_img, (1,2,0))
            print(max(map(max, image_list[i])))
            cv2.imwrite("./data/images_all/image_"+str(count)+".jpg", final_img)
            # images.append(final_img)

            heatmap = np.zeros_like(image_list[0])
            info = gt[i]
            center_y = int(str(info).split(",")[2]) + int(int(str(info).split(",")[4])/2)
            center_x = int(str(info).split(",")[3]) + int(int(str(info).split(",")[5])/2)
            # print(center_x, center_y)
            X1 = np.linspace(1, heatmap.shape[0], heatmap.shape[1])
            Y1 = np.linspace(1, heatmap.shape[0], heatmap.shape[1])
            [X, Y] = np.meshgrid(X1, Y1)
            sigma = 1
            X = X - center_y
            Y = Y - center_x
            D2 = X*X + Y*Y
            E2 = 2.0*sigma*sigma
            Exponent = D2 / E2
            heatmap = np.exp(-Exponent)
            
            print(os.path.exists("./data/images_all"))
            cv2.imwrite("./data/labels_all/heatmap_"+str(count)+".jpg", heatmap)
            count += 1
            # print(count)
            # for x in range(heatmap.shape[0]):
            #     for y in range(heatmap.shape[1]):
            #         heatmap[x, y] = np.exp(-((x-center_x)**2+(y-center_y)**2)/2/sigma**2)

            # gts.append(heatmap)
            # print(int(str(info).split(",")[2]))
            # # center = 
            # plt.imshow(diff_img_list[-1], cmap="gray")
            # plt.show()

    # np.save("images.npy", np.array(images))
    # np.save("labels.npy", np.array(gts))
            
def data_split(imgs, labels):
    images_list = os.listdir(imgs)
    data_num = len(images_list)
    data_list = np.arange(0, data_num)
    offset1 = int(data_num*0.7)
    offset2 = int(data_num*0.8)
    random.shuffle(data_list)
    train_list = data_list[:offset1]
    val_list = data_list[offset1:offset2]
    test_list = data_list[offset2:]

    src_file_list = glob.glob(labels+"/*.jpg")
    for srcfile in src_file_list:
        if int(srcfile.split("_")[2].split(".")[0]) in train_list:
            dst_dir = "./data/labels_train/"
        if int(srcfile.split("_")[2].split(".")[0]) in val_list:
            dst_dir = "./data/labels_val/"
        if int(srcfile.split("_")[2].split(".")[0]) in test_list:
            dst_dir = "./data/labels_test/"
        mycopyfile(srcfile, dst_dir)

    src_file_list = glob.glob(imgs+"/*.jpg")
    # print(src_file_list)
    # print(int(src_file_list[0].split("_")[2].split(".")[0]))
    for srcfile in src_file_list:
        if int(srcfile.split("_")[2].split(".")[0]) in train_list:
            dst_dir = "./data/images_train/"
        if int(srcfile.split("_")[2].split(".")[0]) in val_list:
            dst_dir = "./data/images_val/"
        if int(srcfile.split("_")[2].split(".")[0]) in test_list:
            dst_dir = "./data/images_test/"
        mycopyfile(srcfile, dst_dir)
    
    


def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

    # print(test_list)
    # print(len(val_list))
    # print(len(test_list))



if __name__ == "__main__":
    pre_process()
    # data_split("./data/images_all", "./data/labels_all")