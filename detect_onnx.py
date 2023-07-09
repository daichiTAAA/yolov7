from glob import glob
from pathlib import Path
import time

import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import onnxruntime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# Load model
def load_model(model_path):
    # Set the ONNX Runtime to use all available cores for inference
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = multiprocessing.cpu_count()
    session = onnxruntime.InferenceSession(model_path, options)
    return session

# Run inference
def run_inference(session, img):
    input_name = session.get_inputs()[0].name
    pred_onx = session.run(None, {input_name: img})[0]
    return pred_onx

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Load model
    model_path = 'yolov7.onnx'  # replace with the path to your ONNX model
    session = load_model(model_path)
    
    # Prepare your images
    target_folder_path = './inference/images/test'
    images = glob(target_folder_path + '/*.jpg')
    
    # Create a process pool
    print(f"cpu count: {multiprocessing.cpu_count()}")
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Run inference in parallel
        results = list(executor.map(lambda img: run_inference(session, img), images))
        print(results)
    
    # Now 'results' is a list of inference results for each image
    