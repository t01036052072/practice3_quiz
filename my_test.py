# Inference for ONNX model

import cv2
cnda = True
W = "yolov7-tiny.onnx"
#img = cv2.imread('horses.jpg') #image-based execute!

import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

providers = ['AzureExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InderenceSession(w, providers=providers)
