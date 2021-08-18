from operator import index
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from numpy.core.fromnumeric import mean
from numpy.core.numeric import binary_repr
from numpy.lib.function_base import diff
from numpy.lib.twodim_base import mask_indices
import scipy.signal as signal
import numpy as np
import copy
from scipy import optimize
from scipy.optimize import linear_sum_assignment
import argparse
import torch
from train.center_net import get_pose_net
from PIL import Image
import torchvision.transforms as transforms

class Instance(object):
    def __init__(self, init_px, init_py, idx, frame_id) -> None:
        super(Instance, self).__init__()

        self._init_px = init_px
        self._init_py = init_py
        self._idx = idx

        self._trajectory_frame_id = [frame_id]
        self._trajectory = [[init_px, init_py]]
        self._observed_frame_id = [frame_id]
        self._observed_trajectory = [[init_px, init_py]]
        self._k = None
        self._b = None
        self._active_flag = True
    
    def add_trajectory_frame_id(self, frame_id):
        self._trajectory_frame_id.append(frame_id)
    
    def add_trajectory(self, new_px, new_py):
        self._trajectory.append([new_px, new_py])

    def add_observed_frame_id(self, frame_id):
        self._observed_frame_id.append(frame_id)

    def add_observed_trajectory(self, new_px, new_py):
        self._observed_trajectory.append([new_px, new_py])
        self.update_k_b()
    
    def update_idx(self, new_idx):
        self._idx = new_idx
    
    def update_k_b(self):
        past_len = 60
        if len(self._observed_trajectory) > past_len:
            X_array = np.array(self._observed_trajectory)[-past_len:, 0]
            Y_array = np.array(self._observed_trajectory)[-past_len:, 1]
            ppot, pcov = optimize.curve_fit(func, X_array , Y_array)
            if self._k is None:
                self._k = ppot[0]
                self._b = ppot[1]
            elif abs(ppot[0]-self._k)+abs(ppot[1]-self._b)<1:
                self._k = ppot[0]
                self._b = ppot[1]

            # print(self._k, self._b)

def func(x, k, b):
    return k*x+b

def pic_align(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 20 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)

    goodMatch = matches[:20]
    if len(goodMatch) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
        return H[1, 2], H[0, 2]
    else:
        return [False, False]

def get_distance(pos1, pos2):
    return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)

class MoT(object):
    def __init__(self, image_folder, output_folder, binary_thresh=10, distance_thresh=5, model_path="./checkpoint/model_best.py", use_model=True, output_video_flag=True) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.output_video_flag = output_video_flag
        self.binary_thresh = binary_thresh
        self.distance_thresh = distance_thresh
        self.model_path = model_path
        self.use_model = use_model

        self.model = None
        self.load_model()
        self.model.eval()

        # 图片名称格式应为 000000.jpg 000001.jpg ...
        self.image_path_list = []
        self.image_loader()
        image_0 = np.mean(mpimg.imread(self.image_path_list[0]), axis=2)
        self.background_image_list = []
        self.image_size = [image_0.shape[0], image_0.shape[1]]
        self.total_frame = len(self.image_path_list)
        self.align_flag = 0

        self.instances = []
        self.pre_instances = []
        # self.max_instance_id = 0

        self.before_align_frames = [0]
        self.after_align_frames = [0]
        self.pxs = [0]
        self.pys = [0]

        self.fps = 20
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.output_video_path = os.path.join(output_folder, "video")
        if not os.path.exists(self.output_video_path):
            os.mkdir(self.output_video_path)
        self.video_writer = cv2.VideoWriter(os.path.join(self.output_video_path, 'test.avi'), self.fourcc, self.fps, self.image_size)

        # self.run()

    def load_model(self):
        self.model = get_pose_net()
        state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)['state_dict']
        self.model.load_state_dict(state_dict)

    def image_loader(self):
        # 图片名称格式应为 000000.jpg 000001.jpg
        image_file_list = os.listdir(self.image_folder)
        for i in range(len(image_file_list)):
            image_path = os.path.join(self.image_folder, image_file_list[i])
            # image_gray = np.mean(mpimg.imread(image_path), axis=2)
            # image = cv2.imread(image_path)
            # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.image_path_list.append(image_path)

    def get_image(self, image_path):
        return np.mean(mpimg.imread(image_path), axis=2)
        # return np.mean(Image.open(image_path), axis=2)

    def run(self):
        for i in range(1, self.total_frame):
            print("frame: ", i)
            background_image = self.update_background(i)
            if self.align_flag == 0:
                # print(self.pxs, self.pys)
                # img_Guassian = cv2.GaussianBlur(self.image_list[i],(3,3),0)
                diff_image = np.clip(self.get_image(self.image_path_list[i])-background_image, 0, 255)
                diff_image = signal.medfilt2d(diff_image, (3,3))

                if len(self.background_image_list) == 10 and self.use_model:
                    input_image = []

                    
                    input_image.append(signal.medfilt2d(np.clip(self.get_image(self.image_path_list[i])-self.get_image(self.image_path_list[i-1]), 0, 255), (3,3)))
                    input_image.append(diff_image)
                    input_image.append(self.get_image(self.image_path_list[i]))
                    input_image = np.transpose(np.array(input_image), (1,2,0))
                    
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]
                    )
                    input_image = transform(input_image).unsqueeze(0)
                    input_image = torch.tensor(input_image, dtype=torch.float32)
                    model_output = self.model_pred(input_image)
                    
                    # print(423)
                    model_output = np.uint8(model_output)

                    model_output = cv2.GaussianBlur(model_output, (7,7), 2)

                    model_output = model_output / max(map(max, model_output))# normalize the data to 0 - 1
                    model_output = 255 * model_output
                    # plt.imshow(model_output, cmap="gray")
                    # plt.show()
                    ret, output_binary = cv2.threshold(model_output, max(map(max, model_output))-self.binary_thresh*5, 255, cv2.THRESH_BINARY)
                    
                    kernel = np.ones((3,3), np.uint8)
                    output_binary = cv2.dilate(output_binary, kernel)
                    cv2.imwrite(os.path.join(self.output_folder, "images_net", str(i).zfill(6)+".jpg"), output_binary)
                    # plt.imshow(output_binary, cmap="gray")
                    # plt.show()
                    output_binary = np.uint8(output_binary)
                else:
                    output_binary = np.uint8(np.ones_like(diff_image))

                diff_image = diff_image / max(map(max, diff_image))# normalize the data to 0 - 1
                diff_image = 255 * diff_image # Now scale by 255
                diff_image = diff_image.astype(np.uint8)
                
                cv2.imwrite(os.path.join(self.output_folder, "images_diff", str(i).zfill(6)+".jpg"), diff_image)
                # mean_diff = np.mean(diff_image)
                # diff_image = cv2.threshold(diff_image, mean_diff, 255, cv2.THRESH_TOZERO)
                # # 极值检测方法存疑 假设得到二值化好的图
                # kernel = np.ones((3,3),np.uint8)
                # dilate_diff_image = cv2.dilate(diff_image, kernel, iterations=1)
                # local_extra_image = np.where(diff_image == dilate_diff_image, 255, 0)
                ret, local_extra_image = cv2.threshold(diff_image, 255-self.binary_thresh, 255, cv2.THRESH_BINARY)
                local_extra_image = np.uint8(local_extra_image)
                local_extra_image = local_extra_image*output_binary
                cv2.imwrite(os.path.join(self.output_folder, "diff_binary", str(i).zfill(6)+".jpg"), local_extra_image)
                # cv2.imshow("", diff_image)
                # cv2.waitKey(0)

                self.match_instances(local_extra_image, i)
                self.update_pre_instances(i)
                self.update_instances(i)
                self.generate_output_image(i)
            else:
                print("jump!")

        if self.output_video_flag:
            self.output_video()
    
    def update_background(self, i):
        image_curr = np.uint8(self.get_image(self.image_path_list[i]))
        image_pred = np.uint8(self.get_image(self.image_path_list[i-1]))
        # print(image_curr.dtype)
        px, py = pic_align(image_curr, image_pred)
        if abs(px) < 0.01 and abs(py) < 0.01 and self.align_flag == 0:
            if len(self.background_image_list) < 10:
                self.background_image_list.append(image_pred)
            else:
                self.background_image_list.pop(0)
                self.background_image_list.append(image_pred)
        elif (abs(px) > 0.01 or abs(py) > 0.01) and self.align_flag == 0:
            self.before_align_frames.append(i-1)
            self.background_image_list = []
            self.align_flag = 1
        elif (abs(px) > 0.01 or abs(py) > 0.01) and self.align_flag == 1:
            self.background_image_list = []
            self.align_flag = 1
        elif abs(px) < 0.01 and abs(py) < 0.01 and self.align_flag == 1:
            self.after_align_frames.append(i)
            self.background_image_list.append(image_pred)
            self.align_flag = 0
            # print(self.after_align_frames[-1], self.before_align_frames[-1])
            px, py = pic_align(np.uint8(self.get_image(self.image_path_list[self.after_align_frames[-1]])), np.uint8(self.get_image(self.image_path_list[self.before_align_frames[-1]])))
            self.pxs.append(px)
            self.pys.append(py)

        background_image = np.zeros_like(image_pred)
        for i in range(len(self.background_image_list)):
            background_image += np.uint8(self.background_image_list[i]/(len(self.background_image_list)))

        return background_image

    def model_pred(self, input):
        # print(213)
        with torch.no_grad():
            # print(input)
            output = self.model(input)
            output = output.squeeze().numpy()
            output = np.clip(output, 0, 255)
            output = output / max(map(max, output))# normalize the data to 0 - 1
            output = 255 * output # Now scale by 255
            # 

            return output


    def match_instances(self, local_extra_image, curr_frame):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(local_extra_image, connectivity=8, ltype=None)

        # 得到检测出的候选目标坐标
        detected_centroids = []
        if num_labels < 20:
            for i in range(1, num_labels):
                s = stats[i][-1]
                if s < 30:
                    curr_cen = [int(centroids[i][1]), int(centroids[i][0])]
                    detected_centroids.append(curr_cen)

        # 初始阶段，全部认为是目标
        if len(self.instances) == 0 and len(self.pre_instances) == 0 and len(detected_centroids) > 0:
            for i in range(len(detected_centroids)):
                pos_x = detected_centroids[i][0]
                pos_y = detected_centroids[i][1]
                pos_x += np.sum(self.pxs)
                pos_y += np.sum(self.pys)
                new_pre_ins = Instance(pos_x, pos_y, len(self.pre_instances)+1, curr_frame)
                self.pre_instances.append(new_pre_ins)
                
        elif len(self.instances) == 0 and len(self.pre_instances) > 0 and len(detected_centroids) > 0:
            aligned_idx = []
            N = max(len(self.pre_instances), len(detected_centroids))
            dis_matrix = np.zeros((N, N))
            for i in range(len(self.pre_instances)):
                for j in range(len(detected_centroids)):
                    pos_x = detected_centroids[j][0]
                    pos_y = detected_centroids[j][1]
                    pos_x += np.sum(self.pxs)
                    pos_y += np.sum(self.pys)

                    ins = self.pre_instances[i]
                    use_pred = True
                    if len(ins._observed_trajectory)>30:
                        pred_id = -30
                    else:
                        use_pred = False
                    if use_pred:
                        vx = (ins._observed_trajectory[-1][0] - ins._observed_trajectory[pred_id][0])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                        vy = (ins._observed_trajectory[-1][1] - ins._observed_trajectory[pred_id][1])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                        X_curr = [ins._trajectory[-1][0]+vx, ins._trajectory[-1][1]+vy]
                    else:
                        X_curr = ins._trajectory[-1]

                    dis_matrix[i, j] = get_distance(X_curr, [pos_x, pos_y])
            
            row_ind, col_ind = linear_sum_assignment(dis_matrix)
            for i in range(len(row_ind)):
                if col_ind[i] < len(detected_centroids) and row_ind[i] < len(self.pre_instances):
                    if dis_matrix[row_ind[i], col_ind[i]] < self.distance_thresh and ins._active_flag:
                        aligned_idx.append(col_ind[i])
                        ins = self.pre_instances[row_ind[i]]
                        pos_x = detected_centroids[col_ind[i]][0]
                        pos_y = detected_centroids[col_ind[i]][1]
                        pos_x += np.sum(self.pxs)
                        pos_y += np.sum(self.pys)
                        if self.check_direction(ins, [pos_x, pos_y], curr_frame):
                            ins.add_trajectory_frame_id(curr_frame)
                            ins.add_trajectory(pos_x, pos_y)
                            ins.add_observed_frame_id(curr_frame)
                            ins.add_observed_trajectory(pos_x, pos_y)
                        self.pre_instances[row_ind[i]] = ins
            
            for i in range(len(detected_centroids)):
                if i not in aligned_idx:
                    pos_x = detected_centroids[i][0]
                    pos_y = detected_centroids[i][1]
                    pos_x += np.sum(self.pxs)
                    pos_y += np.sum(self.pys)
                    new_pre_ins = Instance(pos_x, pos_y, len(self.pre_instances)+1, curr_frame)
                    self.pre_instances.append(new_pre_ins)

        elif len(self.instances) > 0 and len(self.pre_instances) > 0 and len(detected_centroids) > 0:
            aligned_idx = []
            N = max(len(self.pre_instances)+len(self.instances), len(detected_centroids))
            dis_matrix = np.zeros((N, N))
            for i in range(len(self.instances)+len(self.pre_instances)):
                for j in range(len(detected_centroids)):
                    pos_x = detected_centroids[j][0]
                    pos_y = detected_centroids[j][1]
                    pos_x += np.sum(self.pxs)
                    pos_y += np.sum(self.pys)

                    if i < len(self.instances):
                        ins = self.instances[i]
                    else:
                        ins = self.pre_instances[i-len(self.instances)]

                    use_pred = True
                    if len(ins._observed_trajectory)>30:
                        pred_id = -30
                    else:
                        use_pred = False
                    if use_pred:
                        vx = (ins._observed_trajectory[-1][0] - ins._observed_trajectory[pred_id][0])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                        vy = (ins._observed_trajectory[-1][1] - ins._observed_trajectory[pred_id][1])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                        X_curr = [ins._trajectory[-1][0]+vx, ins._trajectory[-1][1]+vy]
                    else:
                        X_curr = ins._trajectory[-1]
                    
                    dis_matrix[i, j] = get_distance(X_curr, [pos_x, pos_y])

            row_ind, col_ind=linear_sum_assignment(dis_matrix)
            instances_len = copy.deepcopy(len(self.instances))
            for i in range(len(row_ind)):
                if col_ind[i] < len(detected_centroids) and row_ind[i] < len(self.instances)+len(self.pre_instances) and ins._active_flag:
                    if dis_matrix[row_ind[i], col_ind[i]] < self.distance_thresh:
                        aligned_idx.append(col_ind[i])
                        pos_x = detected_centroids[col_ind[i]][0]
                        pos_y = detected_centroids[col_ind[i]][1]
                        pos_x += np.sum(self.pxs)
                        pos_y += np.sum(self.pys)
                        if row_ind[i] < len(self.instances):
                            ins = self.instances[row_ind[i]]
                            if self.check_direction(ins, [pos_x, pos_y], curr_frame):
                                ins.add_trajectory_frame_id(curr_frame)
                                ins.add_trajectory(pos_x, pos_y)
                                ins.add_observed_frame_id(curr_frame)
                                ins.add_observed_trajectory(pos_x, pos_y)
                            self.instances[row_ind[i]] = ins
                        else:
                            ins = self.pre_instances[row_ind[i]-instances_len]
                            if self.check_direction(ins, [pos_x, pos_y], curr_frame):
                                ins.add_trajectory_frame_id(curr_frame)
                                ins.add_trajectory(pos_x, pos_y)
                                ins.add_observed_frame_id(curr_frame)
                                ins.add_observed_trajectory(pos_x, pos_y)
                            self.pre_instances[row_ind[i]-instances_len] = ins

            for i in range(len(detected_centroids)):
                if i not in aligned_idx:
                    pos_x = detected_centroids[i][0]
                    pos_y = detected_centroids[i][1]
                    pos_x += np.sum(self.pxs)
                    pos_y += np.sum(self.pys)
                    new_pre_ins = Instance(pos_x, pos_y, len(self.pre_instances)+1, curr_frame)
                    self.pre_instances.append(new_pre_ins)

        elif len(self.instances) > 0 and len(self.pre_instances) == 0 and len(detected_centroids) > 0:
            aligned_idx = []
            N = max(len(self.instances), len(detected_centroids))
            dis_matrix = np.zeros((N, N))
            for i in range(len(self.instances)):
                for j in range(len(detected_centroids)):
                    pos_x = detected_centroids[j][0]
                    pos_y = detected_centroids[j][1]
                    pos_x += np.sum(self.pxs)
                    pos_y += np.sum(self.pys)

                    ins = self.instances[i]
                    use_pred = True
                    if len(ins._observed_trajectory)>30:
                        pred_id = -30
                    else:
                        use_pred = False
                    if use_pred:
                        vx = (ins._observed_trajectory[-1][0] - ins._observed_trajectory[pred_id][0])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                        vy = (ins._observed_trajectory[-1][1] - ins._observed_trajectory[pred_id][1])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                        X_curr = [ins._trajectory[-1][0]+vx, ins._trajectory[-1][1]+vy]
                    else:
                        X_curr = ins._trajectory[-1]

                    dis_matrix[i, j] = get_distance(X_curr, [pos_x, pos_y])
            
            row_ind, col_ind=linear_sum_assignment(dis_matrix)
            for i in range(len(row_ind)):
                if row_ind[i] < len(self.instances) and col_ind[i] < len(detected_centroids):
                    if dis_matrix[row_ind[i], col_ind[i]] < self.distance_thresh:
                        aligned_idx.append(col_ind[i])
                        ins = self.instances[row_ind[i]]
                        pos_x = detected_centroids[col_ind[i]][0]
                        pos_y = detected_centroids[col_ind[i]][1]
                        pos_x += np.sum(self.pxs)
                        pos_y += np.sum(self.pys)
                        if self.check_direction(ins, [pos_x, pos_y], curr_frame) and ins._active_flag:
                            ins.add_trajectory_frame_id(curr_frame)
                            ins.add_trajectory(pos_x, pos_y)
                            ins.add_observed_frame_id(curr_frame)
                            ins.add_observed_trajectory(pos_x, pos_y)
                        self.instances[row_ind[i]] = ins
            
            for i in range(len(detected_centroids)):
                if i not in aligned_idx:
                    pos_x = detected_centroids[i][0]
                    pos_y = detected_centroids[i][1]
                    pos_x += np.sum(self.pxs)
                    pos_y += np.sum(self.pys)
                    new_pre_ins = Instance(pos_x, pos_y, len(self.pre_instances)+1, curr_frame)
                    self.pre_instances.append(new_pre_ins)

        if len(self.instances) > 0:
            new_instances = []
            for ins in self.instances:
                if not ins._trajectory_frame_id[-1] == curr_frame and ins._active_flag:
                    if len(ins._observed_frame_id) > 60:
                        pred_id = -60
                    else:
                        pred_id = -10
                    vx = (ins._observed_trajectory[-1][0] - ins._observed_trajectory[pred_id][0])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                    vy = (ins._observed_trajectory[-1][1] - ins._observed_trajectory[pred_id][1])/(ins._observed_frame_id[-1]-ins._observed_frame_id[pred_id])
                    X_curr = [ins._trajectory[-1][0]+vx*(curr_frame-ins._trajectory_frame_id[-1]), ins._trajectory[-1][1]+vy*(curr_frame-ins._trajectory_frame_id[-1])]
                    if ins._k is not None:
                        # print(X_curr)
                
                        y1 = ins._k*X_curr[0]+ins._b
                        x2 = (X_curr[1]-ins._b)/ins._k
                        X_curr_new = [int((X_curr[0]+x2)/2), int((X_curr[1]+y1)/2)]
                        # print(X_curr_new)
                        X_curr = X_curr_new
                    ins.add_trajectory_frame_id(curr_frame)
                    ins.add_trajectory(int(X_curr[0]), int(X_curr[1]))
                new_instances.append(ins)
            
            self.instances = new_instances

    def check_direction(self, ins, pos, curr_frame):
        if len(ins._trajectory) < 2:
            return True
        elif ins._k is None:
            new_vx = (pos[0] - ins._trajectory[-1][0])/(curr_frame-ins._trajectory_frame_id[-1])
            new_vy = (pos[1] - ins._trajectory[-1][1])/(curr_frame-ins._trajectory_frame_id[-1])
            ins_vx = (ins._trajectory[-1][0] - ins._trajectory[0][0])/(ins._trajectory_frame_id[-1]-ins._trajectory_frame_id[0])
            ins_vy = (ins._trajectory[-1][1] - ins._trajectory[0][1])/(ins._trajectory_frame_id[-1]-ins._trajectory_frame_id[0])
            if get_distance(ins._trajectory[-1], pos)<get_distance(ins._trajectory[-1], ins._trajectory[0])/(ins._trajectory_frame_id[-1]-ins._trajectory_frame_id[0])/2 or self.check_v([ins_vx, ins_vy], [new_vx, new_vy])<0.5:
                return False
            else:
                return True
        else:
            new_vx = (pos[0] - ins._trajectory[-1][0])/(curr_frame-ins._trajectory_frame_id[-1])
            new_vy = (pos[1] - ins._trajectory[-1][1])/(curr_frame-ins._trajectory_frame_id[-1])
            ins_vx = (ins._trajectory[-1][0] - ins._trajectory[0][0])/(ins._trajectory_frame_id[-1]-ins._trajectory_frame_id[0])
            ins_vy = (ins._trajectory[-1][1] - ins._trajectory[0][1])/(ins._trajectory_frame_id[-1]-ins._trajectory_frame_id[0])
            if ins._k*pos[0] + ins._b - pos[1] < 2:
                if get_distance(ins._trajectory[-1], pos)<get_distance(ins._trajectory[-1], ins._trajectory[-30])/(ins._trajectory_frame_id[-1]-ins._trajectory_frame_id[-30])/2 or self.check_v([ins_vx, ins_vy], [new_vx, new_vy])<0.5:
                    return False
                else:
                    return True
            else:
                return False

    def check_v(self, ins_v, new_v):
        x = np.array(ins_v)
        y = np.array(new_v)
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)

        return num / denom

    def update_pre_instances(self, curr_frame):
        new_pre_instances = []
        for pre_ins in self.pre_instances:
            if curr_frame-pre_ins._observed_frame_id[-1] > 10:
                pass
            elif len(pre_ins._observed_frame_id) >= 10:
                max_idx = len(self.instances)+1
                pre_ins.update_idx(max_idx)
                self.instances.append(pre_ins)
            else:
                new_pre_instances.append(pre_ins)
        max_idx = 1
        self.pre_instances = []
        for new_pre_ins in new_pre_instances:
            new_pre_ins.update_idx(max_idx)
            self.pre_instances.append(new_pre_ins)
            max_idx += 1

    def update_instances(self, curr_frame):
        new_instances = []
        for ins in self.instances:
            if curr_frame-ins._observed_frame_id[-1] > 60 and ins._active_flag:
                ins._active_flag = False
            new_instances.append(ins)

        self.instances = new_instances


    def generate_output_image(self, curr_frame):
        origin_image = self.get_image(self.image_path_list[curr_frame])
        for ins in self.instances:
            if ins._trajectory_frame_id[-1] == curr_frame:
                pos_x = ins._trajectory[-1][0]
                pos_y = ins._trajectory[-1][1]
                pos_x -= np.sum(self.pxs)
                pos_y -= np.sum(self.pys)
                # print(pos_x, pos_y)
                x_bound = self.image_size[0]
                y_bound = self.image_size[1]
                pos_x = int(pos_x)
                pos_y = int(pos_y)
                if 0 < pos_x < x_bound and 0 < pos_y < y_bound:
                    if ins._observed_frame_id[-1] == curr_frame:
                        color = 255
                    else:
                        color = 0
                    origin_image[min(pos_x, x_bound), max(0, pos_y-5):min(pos_y+6, y_bound)] = color
                    origin_image[max(0, pos_x-5):min(pos_x+6, x_bound), min(pos_y, y_bound)] = color
                    text = str(ins._idx) 
                    cv2.putText(origin_image, text, (pos_y+5, pos_x+5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (color, 0, 0), 1)
        
        cv2.imwrite(os.path.join(self.output_folder, "images", str(curr_frame).zfill(6)+".jpg"), origin_image)

    def output_video(self):
        # output_images = os.listdir(self.output_folder)
        for i in range(self.total_frame):
            frame = cv2.imread(os.path.join(self.output_folder, "images", str(i).zfill(6)+'.jpg'))
            self.video_writer.write(frame)
        self.video_writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default="./data/images/video100/image", help='Images folder')
    parser.add_argument('--output_dir', type=str, default="./output", help='output folder')
    parser.add_argument('--binary_thresh', type=int, default=100, help='Binary threshold')
    parser.add_argument('--distance_thresh', type=int, default=3, help='Distance threshold')
    parser.add_argument('--model_path', type=str, default="./checkpoint/model_best.pth", help='model_path')
    parser.add_argument('--use_model', type=bool, default=False, help="use model flag")
    args = parser.parse_args()
    # 0 13 16 
    # mot = MoT(image_folder="./data/images/video21/image", output_folder="./output")
    mot = MoT(image_folder=args.images_dir, output_folder=args.output_dir, binary_thresh=args.binary_thresh, distance_thresh=args.distance_thresh, model_path=args.model_path, use_model=args.use_model)
    mot.run()