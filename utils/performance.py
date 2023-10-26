import os
import numpy as np
import pandas as pd
import cv2
import torch
import ultralytics
from ultralytics import YOLO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import display
import ultralytics.utils.plotting as plotting

from pytube import YouTube
from collections import defaultdict


DATA_PATH = "./datasets/COCO/val2017"

def warming_up(model, data_path):
    img_names = os.listdir(data_path)
    # Прогрев
    for i in range(10):
        img = cv2.imread(os.path.join(data_path, img_names[i]))
        model.predict(img, device="0", imgsz = 640, conf=0.5, verbose=False)

def detection_performance(data_path, models_versions: str, batch_sizes: list, count_img_inference: int, model_config: dict):
    img_names = list(filter(lambda file: file.endswith(".jpg") or file.endswith(".jpeg"), os.listdir(data_path)))
    if (len(img_names) < count_img_inference):
        count_img = len(img_names)
    else:
        count_img = count_img_inference

    benchmark_df = pd.DataFrame(columns=["model", "batch_size", "prepocess_time",
                                "inference_time", "postprocess_time", "total_time",
                                "avg_fps"])
    rand_img_idx = np.arange(len(img_names))
    np.random.shuffle(rand_img_idx)

    for model_size_version in models_versions:
        model_name = f"yolov8{model_size_version}.pt"
        model = YOLO(model_name)
        print(f"Process {model_name}")

        warming_up(model, data_path)
        
        for batch_size in batch_sizes:
            total_preprocess = 0
            total_inference = 0
            total_postprocess = 0
            processed_img_count = 0
            if (count_img // batch_size == 0):
                continue
            for i in range((count_img - batch_size) // batch_size):
                batch_img = []
                for j in range(batch_size):
                    try:
                        img = cv2.imread(os.path.join(data_path, img_names[rand_img_idx[i * batch_size + j]]))
                    except IndexError as ex:
                        print(f"Count img {count_img}")
                        print(f"Batch size {batch_size}")
                        print(f"i = {i}")
                        print(f"i = {j}")
                        raise ex

                    processed_img_count += 1
                    batch_img.append(img)
                
                predicts = model.predict(batch_img, **model_config)
                for result in predicts:
                    total_preprocess += result.speed["preprocess"]
                    total_inference += result.speed["inference"]
                    total_postprocess += result.speed["postprocess"]
            
            total_time = total_preprocess + total_inference + total_postprocess
            df = pd.DataFrame({
                "model": model_name, 
                "batch_size": batch_size, 
                "prepocess_time": total_preprocess / processed_img_count, 
                "inference_time": total_inference / processed_img_count, 
                "postprocess_time": total_postprocess / processed_img_count, 
                "total_time": total_time / processed_img_count,
                "avg_fps": processed_img_count / total_time * 1000
            }, index=[0])
            benchmark_df = pd.concat([benchmark_df, df], ignore_index = True)
    return benchmark_df

def video_performance_mot(video_path, models_versions: str, model_config: dict, max_img: int = None):
    benchmark_df = pd.DataFrame(columns=["model", "prepocess_time",
                                "inference_time", "postprocess_time", "total_time",
                                "avg_fps"])
    
    for model_size_version in models_versions:
        model_name = f"yolov8{model_size_version}.pt"
        model = YOLO(model_name)
        print(f"Process {model_name}")
        
        total_preprocess = 0
        total_inference = 0
        total_postprocess = 0

        cap = cv2.VideoCapture(video_path)
        processed_img_count = 0
        if (max_img is None):
            max_img = np.inf
        while cap.isOpened() and processed_img_count < max_img:
            success, frame = cap.read()

            if success:
                results = model.track(frame, persist=True, **model_config)
                for result in results:
                    total_preprocess += result.speed["preprocess"]
                    total_inference += result.speed["inference"]
                    total_postprocess += result.speed["postprocess"]

                processed_img_count += 1
            else:
                break
        cap.release()

        total_time = total_preprocess + total_inference + total_postprocess
        df = pd.DataFrame({
            "model": model_name, 
            "prepocess_time": total_preprocess / processed_img_count, 
            "inference_time": total_inference / processed_img_count, 
            "postprocess_time": total_postprocess / processed_img_count, 
            "total_time": total_time / processed_img_count,
            "avg_fps": processed_img_count / total_time * 1000
        }, index=[0])
        benchmark_df = pd.concat([benchmark_df, df], ignore_index = True)
    
    return benchmark_df


