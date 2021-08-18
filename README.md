这一版本输入为顺序的图像帧，每个视频所有图像帧保存在相应文件夹中，图像名为000000.jpg格式
输出默认有图像和视频两种格式，分别保存在output文件夹中两个子文件夹下


# 激活环境
conda activate FairMOT

# 运行程序
python mot_rocket.py --images_dir ./data/video_x/image --output_dir ./output --model_path ./checkpoint/model_best.pth


# 可选参数以及默认值如下
parser.add_argument('--images_dir', type=str, default="./data/images/video21/image", help='Images folder')
parser.add_argument('--output_dir', type=str, default="./output", help='output folder')
parser.add_argument('--binary_thresh', type=int, default=10, help='Binary threshold')
parser.add_argument('--distance_thresh', type=int, default=5, help='Distance threshold')
parser.add_argument('--model_path', type=str, default="./checkpoint/model_best.pth", help='model_path')
parser.add_argument('--use_model', type=bool, default=True, help="use model flag")
