import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
from keras.preprocessing.sequence import pad_sequences # type: ignore

sys.path.append(r"C:\Users\USER\Desktop\CityU\CityU_CS\CS4514 Project\utils\openpose\bin\python\openpose\Release")
import pyopenpose as op # type: ignore

if __name__ == "__main__":
    try:
        frame_dir = r"E:\dataset\tvb-hksl-news\frames"
        keypoint_dir = r"E:\dataset\tvb-hksl-news\keypoints_openpose"

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../models/"
        params["num_gpu"] = 1
        numberGPUs = int(params["num_gpu"])

        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        date_list = os.listdir(frame_dir)
        date_list.sort()

        for date in date_list:
            name_list = os.listdir(os.path.join(frame_dir, date))
            name_list.sort()
            for name in name_list:
                if os.path.exists(os.path.join(keypoint_dir, date, f"{name}.npy")):
                    continue
                name_path = os.path.join(frame_dir, date, name)
                imagePaths = op.get_images_on_directory(name_path)
                # process images
                for imageBaseId in range(0, len(imagePaths), numberGPUs):
                    images = []
                    for gpuId in range(numberGPUs):
                        imageId = imageBaseId + gpuId
                        if imageId < len(imagePaths):
                            image = cv2.imread(imagePaths[imageId])
                            images.append(image)
                    datum = op.Datum()
                    datum.cvInputData = images
                    opWrapper.emplaceAndPop([datum])
                    keypoints_array = []
                    for i in range(len(datum.poseKeypoints)):
                        keypoints = datum.poseKeypoints[i].flatten() if datum.poseKeypoints[i] is not None else np.zeros(1)
                        keypoints_array.append(keypoints)
                    keypoints_array = np.array(keypoints_array)
                    max_length = max([len(sequence) for sequence in keypoints_array])
                    keypoints_array = pad_sequences(keypoints_array, maxlen=max_length, padding="post")
                    
                    os.makedirs(os.path.join(keypoint_dir, date), exist_ok=True)
                    np.save(os.path.join(keypoint_dir, date, f"{name}.npy"), keypoints_array)

    except Exception as e:
        print(e)
        sys.exit(-1)