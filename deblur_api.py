import os
from flask import Flask, request, redirect, url_for, send_from_directory, make_response, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO
from werkzeug.exceptions import BadRequest
from model import DEBLUR
import argparse
import scipy.misc
from PIL import Image
import io
import tensorflow as tf
#calculate laplacian
import numpy as np
import cv2
import math
# AVX2's warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='test', help='determine whether train or test')
    parser.add_argument('--datalist', type=str, default='./datalist_gopro.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=256,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=256,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    ## Input file
    parser.add_argument('--input_path', type=str, default='./input',
                        help='input path for testing images')
    ## Output file
    parser.add_argument('--output_path', type=str, default='./output',
                        help='output path for testing images')
    parser.add_argument('--img_dir', type=str, default='./input',
                        help='input path for testing images')
    args = parser.parse_args()
    return args

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['image']

        ## 將圖片以二進制寫入內存中，防止將圖片放入文件夾中，占用大量磁碟
        inputIO = io.BytesIO()
        outputIO = io.BytesIO()
        image.save(inputIO)

        ## deblur test image
        args = parse_args()
        deblur = DEBLUR(args)
        output = deblur.test(inputIO)
        output.save(outputIO, "png")

        img_str = outputIO.getvalue()
        ## 把二進制作為response發回前端，並設置首部字段
        response = make_response(img_str)
        response.headers.set('Content-Type', 'image/png')

    return response



if __name__ == '__main__':

    app.run('localhost', port=5555, debug=True)
