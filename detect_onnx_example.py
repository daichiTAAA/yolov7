from glob import glob
import os
import random
import requests
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

cuda = True
export_model_path = "./yolov7.onnx"

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(export_model_path, providers=providers)

# GPUを使用する場合
# Change to True when onnxruntime (like onnxruntime-gpu 1.0.0 ~ 1.1.2) cannot be imported.
# add_cuda_path = False

# For Linux, see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup
# Below is example for Windows
# if add_cuda_path:
#     cuda_dir = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin'
#     cudnn_dir = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\libnvvp'
#     if not (os.path.exists(cuda_dir) and os.path.exists(cudnn_dir)):
#         raise ValueError("Please specify correct path for CUDA and cuDNN. Otherwise onnxruntime cannot be imported.")
#     else:
#         if cuda_dir == cudnn_dir:
#             os.environ["PATH"] = cuda_dir + ';' + os.environ["PATH"]
#         else:
#             os.environ["PATH"] = cuda_dir + ';' + cudnn_dir + ';' + os.environ["PATH"]

# import psutil
# import onnxruntime
# import numpy

# print(f"onnxruntime version: {onnxruntime.__version__}")
# print(f"os.environ['PATH']: {os.environ['PATH']}")
# print(f"os.environ['CUDA_PATH']: {os.environ['CUDA_PATH']}")
# print(f"onnxruntime.get_available_providers() : {onnxruntime.get_available_providers()}")
# assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
# device_name = 'gpu'

# sess_options = onnxruntime.SessionOptions()

# Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
# Note that this will increase session creation time so enable it for debugging only.
# sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_{}.onnx".format(device_name))

# Please change the value according to best setting in Performance Test Tool result.
# sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

# session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CUDAExecutionProvider'])
# session = onnxruntime.InferenceSession(export_model_path, providers=['CUDAExecutionProvider'])



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

start_time = time.time()

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

# Prepare your images
target_folder_path = './inference/images/test'
img_paths = glob(target_folder_path + '/*.jpg')
# runs/detectフォルダ中のフォルダのリストを確認し、exp+連番+1となるよう、画像をruns/detectの下のexp+連番のフォルダに保存
save_folder_parent_path = './runs/detect'
exp_folders = glob(save_folder_parent_path + '/exp*')
exp_nums = [int(exp_folder.split('\\')[-1].replace('exp','')) for exp_folder in exp_folders if exp_folder.split('\\')[-1].replace('exp','').isdigit()]
print(f"exp_nums: {exp_nums}")
if exp_nums == []:
    save_folder_num = 0
else:
    # exp_numsからint以外を除外する
    exp_nums = [int(exp_num) for exp_num in exp_nums if isinstance(exp_num, int)]
    # exp_numsの最大値を取得し、+1する
    save_folder_num = max(exp_nums) + 1
print(f"save_folder_parent_path: {save_folder_parent_path}")
save_folder_path = save_folder_parent_path + '/exp' + str(save_folder_num)
if not Path(save_folder_path).exists():
    Path(save_folder_path).mkdir(parents=True, exist_ok=True)

# print(img_paths)
for img_path in img_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    
    im = image.astype(np.float32)
    im /= 255
    im.shape
    
    outname = [i.name for i in session.get_outputs()]
    outname
    
    inname = [i.name for i in session.get_inputs()]
    inname
    
    inp = {inname[0]:im}

    outputs = session.run(outname, inp)[0]
    print(outputs)

    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
    
    save_image_path = save_folder_path + '/' + img_path.split('\\')[-1]
    print(f"save_image_path: {save_image_path}")
    Image.fromarray(ori_images[0]).save(save_image_path)

end_time = time.time()
execution_time = end_time - start_time
print('Execution Time:', execution_time)