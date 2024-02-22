import os
import pickle

import eval
from flask import Flask, jsonify, render_template, request
import json
import torch
import numpy as np

# -*- coding: utf-8 -*-
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from gevent import pywsgi
import json

from utils.data_utils import save_dataset

app = Flask(__name__)  # 实例化app对象
CORS(app, resources={r"/*": {"origins": "*"}})
testInfo = {}


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # 允许这个域名访问
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/api/data/TSP', methods=['GET', 'POST'])
def get_data_tsp():
    my_array = request.json
    print('Received:', my_array)
    with open("data/cifar10_demo1_labels.pkl", "rb") as fl:
        labels = pickle.load(fl)
    with open("data/cifar10_demo1_tsne.pkl", "rb") as f1:
        a = pickle.load(f1)
    if len(my_array)>1:
        re_arry = []
        for w in range(len(my_array)):
            re_arry.append(a[my_array[w][0], :].tolist())
        save_dataset(re_arry, f'data/linshifanhuiwenjian/received.pkl')

        os.system("python eval.py --model %s -f --run_mode %s --dataset %s --dataset_number %d --sample_size %d" % ('pretrained/TSP', 'test', 'demo', 1, len(my_array)))
        with open('data/linshifanhuiwenjian/1.pkl', 'rb') as f1:
            order = pickle.load(f1)
        with open('data/linshifanhuiwenjian/2.pkl', 'rb') as f2:
            dist = pickle.load(f2)

        tem0 = []
        sum1 = np.max(dist)
        print(order)
        colors = ['#00BFFF', '#00FF00', '#FFA500', '#FF0000', '#800080', '#00FFFF', '#FF69B4', '#006400', '#A52A2A',
                  '#00CED1']
        for i in range(len(order)):
            tem1 = []
            tem1.append(my_array[order[i]][0])
            tem1.append(round((dist[i]/sum1)*1980))
            if int(labels[my_array[order[i]][0]]) <= -1:
                tem1.append(0)
                tem1.append(colors[int(labels[my_array[order[i]][0]])])
                tem1.append(265+50)
            else:
                tem1.append(int(labels[my_array[order[i]][0]]))
                tem1.append(colors[int(labels[my_array[order[i]][0]])])
                tem1.append(5)

            tem0.append(tem1)

        print('Return:', tem0)
        return jsonify(tem0)
    else:
        return 'Len of input < 2.'

@app.route('/api/data/SM', methods=['GET', 'POST'])
def get_data_sm():
    my_array = request.json
    print('Received:', my_array)
    with open("data/cifar10_demo1_labels.pkl", "rb") as fl:
        labels = pickle.load(fl)
    with open("data/cifar10_demo1_tsne.pkl", "rb") as f1:
        a = pickle.load(f1)
    if len(my_array)>1:
        re_arry = []
        for w in range(len(my_array)):
            re_arry.append(a[my_array[w][0], :].tolist())
        save_dataset(re_arry, f'data/linshifanhuiwenjian/received.pkl')

        os.system("python eval.py --model %s -f --run_mode %s --dataset %s --dataset_number %d --sample_size %d" % ('pretrained/stress', 'greedy', 'test', 'demo', 1, len(my_array)))
        with open('data/linshifanhuiwenjian/1.pkl', 'rb') as f1:
            order = pickle.load(f1)
        with open('data/linshifanhuiwenjian/2.pkl', 'rb') as f2:
            dist = pickle.load(f2)

        tem0 = []
        sum1 = np.max(dist)
        print(order)
        colors = ['#00BFFF', '#00FF00', '#FFA500', '#FF0000', '#800080', '#00FFFF', '#FF69B4', '#006400', '#A52A2A',
                  '#00CED1']
        for i in range(len(order)):
            tem1 = []
            tem1.append(my_array[order[i]][0])
            tem1.append(round((dist[i]/sum1)*1980))
            if int(labels[my_array[order[i]][0]]) <= -1:
                tem1.append(0)
                tem1.append(colors[int(labels[my_array[order[i]][0]])])
                tem1.append(265+50)
            else:
                tem1.append(int(labels[my_array[order[i]][0]]))
                tem1.append(colors[int(labels[my_array[order[i]][0]])])
                tem1.append(5)

            tem0.append(tem1)

        print('Return:', tem0)
        return jsonify(tem0)
    else:
        return 'Len of input < 2.'


@app.route('/api/data/MI', methods=['GET', 'POST'])
def get_data_mi():
    my_array = request.json
    print('Received:', my_array)
    with open("data/cifar10_demo1_labels.pkl", "rb") as fl:
        labels = pickle.load(fl)
    with open("data/cifar10_demo1_tsne.pkl", "rb") as f1:
        a = pickle.load(f1)
    if len(my_array)>1:
        re_arry = []
        for w in range(len(my_array)):
            re_arry.append(a[my_array[w][0], :].tolist())
        save_dataset(re_arry, f'data/linshifanhuiwenjian/received.pkl')

        os.system("python eval.py --model %s -f --run_mode %s --dataset %s --dataset_number %d --sample_size %d" % ('pretrained/MI', 'test', 'demo', 1, len(my_array)))
        with open('data/linshifanhuiwenjian/1.pkl', 'rb') as f1:
            order = pickle.load(f1)
        with open('data/linshifanhuiwenjian/2.pkl', 'rb') as f2:
            dist = pickle.load(f2)

        tem0 = []
        sum1 = np.max(dist)
        print(order)
        colors = ['#00BFFF', '#00FF00', '#FFA500', '#FF0000', '#800080', '#00FFFF', '#FF69B4', '#006400', '#A52A2A',
                  '#00CED1']
        for i in range(len(order)):
            tem1 = []
            tem1.append(my_array[order[i]][0])
            tem1.append(round((dist[i]/sum1)*1980))
            if int(labels[my_array[order[i]][0]]) <= -1:
                tem1.append(0)
                tem1.append(colors[int(labels[my_array[order[i]][0]])])
                tem1.append(265+50)
            else:
                tem1.append(int(labels[my_array[order[i]][0]]))
                tem1.append(colors[int(labels[my_array[order[i]][0]])])
                tem1.append(5)

            tem0.append(tem1)

        print('Return:', tem0)
        return jsonify(tem0)
    else:
        return 'Len of input < 2.'

@app.route('/my-endpoint', methods=['POST'])
def my_endpoint():
    data = request.json
    print(data)
    return 'Received data'

@app.route('/myFlaskEndpoint', methods=['POST'])
def my_view_function():
    my_array = request.json
    print(my_array)
    return jsonify(my_array)

if __name__ == '__main__':
    # app.run()
    server = pywsgi.WSGIServer(('127.0.0.1', 7000), app)
    server.serve_forever()
