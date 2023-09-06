import json
from sklearn.metrics import ndcg_score
import random
from numpy import inf
from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.order.state_order import StateOrder
from utils.beam_search import beam_search
from sklearn import preprocessing
import options
from torchvision import datasets, transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from torch import nn
from sklearn.cluster import KMeans
import torch.nn.functional as F

from utils.data_utils import save_dataset


def moransi(permuted):
    row_col = len(permuted[0, :])
    N = pow(row_col, 2)
    W = (row_col - 2) * (row_col - 2) * 4 + (row_col - 2) * 3 * 2 + (row_col - 2) * 3 * 2 + 8

    meank = 0
    for i in permuted:
        for j in i:
            meank += j
    num = 0
    denom = 0
    for i in range(row_col):
        for j in range(row_col):
            denom += pow(permuted[i, j] - meank/N, 2)
            innersum = 0
            for y in range(max(0, i-1), min(row_col, i+2)):
                for x in range(max(0, j - 1), min(row_col, j + 2)):
                    if y != i or x !=j:
                        if j - x >= -1 and j - x <= 1 and i == y:
                            innersum += (permuted[i, j] * N - meank) * (permuted[y, x] * N - meank)
                        if j == x and i - y >= -1 and i - y <= 1:
                            innersum += (permuted[i, j] * N - meank) * (permuted[y, x] * N - meank)
            num += innersum
    if num == 0 and denom == 0:
        return 1

    # return -(((N/W) * (num/denom))/(N*N) - N + 2)/2
    return ((N/W) * (num/denom))/(N*N)

def real_distance_stress(dis):

    newdis = torch.zeros(dis.size(0))
    new_d = torch.zeros(dis.size(0), dis.size(0))
    for i in range(dis.size(0) - 1):
        newdis[i+1] = dis[i, i + 1]

    for i in range(dis.size(0)-1):
        for j in range(i+1, dis.size(0)):
            new_d[i, j] = newdis[i+1:j+1].sum()
    return new_d.cuda()

def target_distance_stress(X):
    return torch.norm(X[:, None] - X, dim=2, p=2).triu()

def eucl_distance_stress(X):
    return torch.norm(X[:, None] - X, dim=2, p=2)

def linear_solver(pi, D, alpha=1):

    A = torch.zeros(D.size(0), D.size(0)).cuda()
    w = torch.pow(D, -alpha).cuda()
    w = torch.where(torch.isinf(w), torch.full_like(w, 0), w)
    pi_diff = pi.unsqueeze(0) - pi.unsqueeze(1)

    A[:, :] = torch.where(pi_diff < 0, w, -w)
    b = (w*D).view(-1).unsqueeze(1)
    row1 = torch.zeros(D.size(0)*D.size(0)).cuda()
    row2 = torch.zeros(D.size(0) * D.size(0)).cuda()
    col1 = torch.zeros(D.size(0)*D.size(0)).cuda()
    col2 = torch.zeros(D.size(0) * D.size(0)).cuda()
    values1 = torch.zeros(D.size(0)*D.size(0)).cuda()
    values2 = torch.zeros(D.size(0) * D.size(0)).cuda()
    tem1 = torch.sort(torch.clone(torch.argsort(torch.sort(row1).values))).values
    for i in range(D.size(0)-1):
        row1[i*D.size(0):i*D.size(0)+D.size(0)] = tem1[i*D.size(0)+i+1:i*D.size(0)+i+D.size(0)+1]
        row2 = torch.clone(row1)
        col1[i * D.size(0):i * D.size(0) + D.size(0)-1] = i
        col1[i * D.size(0) + D.size(0)-1] = i+1
        col2[i * D.size(0):i * D.size(0) + D.size(0)] = torch.cat((tem1[0:D.size(0)][i+1:], tem1[0:D.size(0)][0:i+1]), dim=0)
        values1[i*D.size(0):i*D.size(0)+D.size(0)] = torch.cat((A[i, i+1:],A[i+1, :i+1]), dim=0)
        values2[i*D.size(0):i*D.size(0)+D.size(0)] = torch.cat((A[i+1:, i],A[:i+1, i+1]), dim=0)
    row = torch.cat((row1, row2), dim=0).long()
    col = torch.cat((col1, col2), dim=0).long()
    values = torch.cat((values1, values2), dim=0)
    indicies = [row.cpu().numpy(), col.cpu().numpy()]
    das = torch.sparse_coo_tensor(indicies, values, size=[D.size(0)*D.size(0), D.size(0)]).to_dense()
    X = torch.lstsq(b, das).solution
    new_order = torch.argsort(X[0:D.size(0), 0])

    return torch.pow(X[D.size(0):, 0], 2).sum(), new_order, X[0:D.size(0), 0]

def stress(real_distance, target_distance):

    s = 1/torch.pow(target_distance, 2) * torch.pow(real_distance - target_distance, 2)
    s = torch.where(torch.isnan(s), torch.full_like(s, 0), s)
    return s.sum(0).sum()


def calculate_distance(point1, point2):
    # 计算两个点之间的距离，这里我们假设点是二维坐标 (x, y)
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)

def linear_arrangement_from_adjacency_matrix(adj_matrix, order):
    # 根据指定的行列顺序计算线性排列的距离
    graph = {}
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[order[i], order[j]] == 1:
                if (i, j) not in graph:
                    graph[(i, j)] = []
                if (j, i) not in graph:
                    graph[(j, i)] = []
                graph[(i, j)].append((j, i))
                graph[(j, i)].append((i, j))

    total_distance = 0
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            total_distance += calculate_distance(vertex, neighbor)

    return total_distance

class Order(object):

    NAME = 'order'

    @staticmethod
    # def get_costs(dataset, pi, cost_choose=options.get_options().cost_choose):
    def get_costs(dataset, pi, cost_choose='tsp'):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset)).cuda()

        if cost_choose == 'stress':

            ret_res = torch.zeros(d.size(0)).cuda()
            for i in range(pi.size(0)):
                W = d[i, :, :].cuda()

                new_order = pi[i]
                for b in range(2):
                    lossh, new_order, res = linear_solver(new_order, eucl_distance_stress(W))

                node_size = W.size(0)
                dis_1 = torch.zeros(node_size, node_size)
                for a in range(node_size):
                    for b in range(node_size):
                        dis_1[a, b] = abs(res[a] - res[b])

                real_distance = dis_1.triu().cuda()
                target_distance = target_distance_stress(W)
                s1 = stress(real_distance, target_distance)
                if s1 == inf:
                    s1 = 0
                ret_res[i] = s1

            return ret_res, None
        elif cost_choose == 'moransI':

            ret_res = torch.zeros(d.size(0))
            for a in range(d.size(0)):
                ret_res[a] = 1 - moransi(d[a, :, :])
            return ret_res, None

        elif cost_choose == 'n':
            num_batches = 10
            num_nodes = 240
            ret_res = torch.zeros(d.size(0))
            path = 'data/dummy_data/dumpydata_train_y_tem.pkl'
            with open(path, 'rb') as f1:
                data_tsne = pickle.load(f1).reshape(10, 240)

            path1 = 'data/dummy_data/dumpydata_train_i.pkl'
            with open(path1, 'rb') as f:
                true_orders = pickle.load(f)
            print(pi[0], pi.size())
            predicted_orders = torch.zeros(10, 240)
            gt = torch.zeros(10, 240)
            for i in range(10):
                for j in range(240):
                    predicted_orders[i, j] = data_tsne[i, pi[i, j]]
                    gt[i, j] = data_tsne[i, true_orders[i, j]]



            # predicted_orders = data_tsne.gather(1, pi.cpu().unsqueeze(-1).expand_as(dataset)).reshape(10, 240)
            # d1 = data_tsne.gather(1, true_orders.unsqueeze(-1).expand_as(dataset)).reshape(10, 240)
            #
            # print(d1.size(), predicted_orders.size())

            # from allrank.models.metrics import ndcg
            # print(ndcg(predicted_orders, d1))

            # 计算NDCG@5
            ndcg_scores = []
            for i in range(10):
                predicted_order = predicted_orders[i].numpy().tolist()
                true_order = gt[i].numpy().tolist()

                # 使用sklearn的ndcg_score计算NDCG
                ret_res[i] = 1 - ndcg_score([true_order], [predicted_order], k=5)

            return ret_res, None

        elif cost_choose == "LA":
            ret_res = torch.zeros(d.size(0))

            # #Linear arrangement LA (min) 要用顺序计算，不用坐标计算
            path = 'data/sch/sch.json'
            with open(path, "r") as f:
                row_data = json.load(f)
            row_data = torch.Tensor(row_data)
            #
            # with open(f'data/sch/sch.pkl', 'rb') as f1:
            #     row_data = pickle.load(f1)
            #
            # # with open(f'D:/CODE\VON-master\VON\data\sch\sch_bi.pkl', 'rb') as f1:
            # #     row_data = pickle.load(f1)
            #
            graphsize = 242
            ret_res = torch.zeros(d.size(0))
            for a in range(d.size(0)):
                tem1 = 0
                im1 = torch.zeros([graphsize, graphsize])
                new_matrix1 = torch.zeros([graphsize, graphsize])
                im1[:, :] = row_data[a, :, :]
                newim1 = torch.zeros([graphsize, graphsize])
                for k in range(graphsize):
                    new_matrix1[k, :] = im1[pi[a, k], :]
                for k in range(graphsize):
                    newim1[:, k] = new_matrix1[:, pi[a, k]]
                for s in range(graphsize):
                    for d in range(graphsize - s):
                        tem1 += abs(newim1[s, d + s] * (d))
                ret_res[a] = tem1
            return ret_res, None
        elif cost_choose == "BW":
            #Bandwidth BW
            ret_res = torch.zeros(d.size(0))
            for a in range(row_data.size(0)):
                im1 = torch.zeros([graphsize, graphsize])
                new_matrix1 = torch.zeros([graphsize, graphsize])
                im1[:, :] = row_data[a, :, :]
                newim1 = torch.zeros([graphsize, graphsize])

                for k in range(graphsize):
                    new_matrix1[k, :] = im1[pi[a, k], :]
                for k in range(graphsize):
                    newim1[:, k] = new_matrix1[:, pi[a, k]]
                tem1 = 0
                for i in range(row_data.size(2)):
                    te1 = 0
                    for j in range(row_data.size(1) - i):
                        # print(newim1[i, j])
                        if newim1[j + i, i] == 1 and j > te1:
                            te1 = j
                    if tem1 < te1:
                        tem1 = te1
                ret_res[a] = tem1
            return ret_res, None
        elif cost_choose == "PR":
            # #Profile PR
            ret_res = torch.zeros(d.size(0))
            for a in range(row_data.size(0)):
                im1 = torch.zeros([graphsize, graphsize])
                new_matrix1 = torch.zeros([graphsize, graphsize])
                im1[:, :] = row_data[a, :, :]
                newim1 = torch.zeros([graphsize, graphsize])

                for k in range(graphsize):
                    new_matrix1[k, :] = im1[pi[a, k], :]
                for k in range(graphsize):
                    newim1[:, k] = new_matrix1[:, pi[a, k]]
                tem1 = 0
                for i in range(row_data.size(2)):
                    te1 = 0
                    for j in range(row_data.size(1) - i):
                        if newim1[j+i, i] == 1 and j > te1:
                            te1 = j
                    tem1 += te1
                ret_res[a] = tem1
            return ret_res, None
        else:#TSP

            hd = d[0, :, :]
            loc_dis = torch.zeros(hd.size(0))
            plus_tem = 0
            for i in range(hd.size(0) - 1):
                plus_tem += torch.sqrt(pow((hd[i, 0] - hd[i + 1, 0]), 2) + pow((hd[i, 1] - hd[i + 1, 1]), 2)).cpu()
                loc_dis[i + 1] = plus_tem
            save_dataset(loc_dis.tolist(), f'data/linshifanhuiwenjian/2.pkl')

            return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):

        return OrderDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateOrder.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = Order.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class OrderDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=10, offset=0, distribution='None', mode='train', mission='IN', dataset_number=1, epoch=0):
        super(OrderDataset, self).__init__()

        if mission == 'fashion_mnist':
            if mode == 'train':
                with open('data/fashion_mnist/train/fmtofull.pkl', 'rb') as f1:
                # with open('data/fashion_mnist/train/combine_local_full.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne[:size*num_samples].reshape(num_samples, size, 2)
                # if num_samples == 10:
                #     with open('data/fashion_mnist/train/fmtofull.pkl', 'rb') as f1:
                #     # with open('data/fashion_mnist/train/combine_mix_full.pkl', 'rb') as f1:
                #         data_tsne = pickle.load(f1)
                #     self.data = data_tsne[0:num_samples * size].reshape(num_samples, size, 2)
                # else:
                #     with open('data/fashion_mnist/train/combine_mix_full.pkl', 'rb') as f1:
                #         data_tsne = pickle.load(f1)
                #     # with open('data/fashion_mnist/train/combine_local_full.pkl', 'rb') as f1:
                #     #     data_tsne = pickle.load(f1)
                #     self.data = data_tsne[epoch*num_samples * size:epoch*num_samples * size+num_samples * size].reshape(num_samples, size, 2)
            elif mode == 'test':
                test_model = '50g'
                if test_model == '50g':
                    data_tsne = torch.FloatTensor(3, 50, 2)
                    images = torch.FloatTensor(3, 50, 1, 28, 28)
                    for i in range(3):
                        with open(f'data/fashion_mnist/test/50_tsne_data{i + 1}.pkl', 'rb') as f1:
                            data_tsne[i, :, :] = pickle.load(f1)[0]
                        with open(f'data/fashion_mnist/test/50_im_data{i + 1}.pkl', 'rb') as f2:
                            images[i, :, :, :, :] = pickle.load(f2)[0]
                elif test_model == '50l':
                    data_tsne = torch.FloatTensor(10, 50, 2)
                    images = torch.FloatTensor(10, 50, 1, 28, 28)
                    for i in range(10):
                        with open(f'data/fashion_mnist/test/label{i}test.pkl', 'rb') as f1:
                            data_tsne[i, :, :] = pickle.load(f1)
                        with open(f'data/fashion_mnist/test/label{i}test_im.pkl', 'rb') as f2:
                            images[i, :, :, :, :] = pickle.load(f2)
                elif test_model == '100g':
                    data_tsne = torch.FloatTensor(10, 100, 2)
                    images = torch.FloatTensor(10, 100, 1, 28, 28)
                    for i in range(10):
                        with open(f'data/fashion_mnist/test/test_mix{i}_100.pkl', 'rb') as f1:
                            data_tsne[i, :, :] = pickle.load(f1)
                        with open(f'data/fashion_mnist/test/test_im_mix{i}_100.pkl', 'rb') as f2:
                            images[i, :, :, :, :] = pickle.load(f2)
                elif test_model == '100l':
                    data_tsne = torch.FloatTensor(10, 100, 2)
                    images = torch.FloatTensor(10, 100, 1, 28, 28)
                    for i in range(10):
                        with open(f'data/fashion_mnist/test/label{i}test_100.pkl', 'rb') as f1:
                            data_tsne[i, :, :] = pickle.load(f1)
                        with open(f'data/fashion_mnist/test/label{i}test_im_100.pkl', 'rb') as f2:
                            images[i, :, :, :, :] = pickle.load(f2)
                elif test_model == '150g':
                    data_tsne = torch.FloatTensor(10, 150, 2)
                    images = torch.FloatTensor(10, 150, 1, 28, 28)
                    for i in range(10):
                        with open(f'data/fashion_mnist/test/test_mix{i}_150.pkl', 'rb') as f1:
                            data_tsne[i, :, :] = pickle.load(f1)
                        with open(f'data/fashion_mnist/test/test_im_mix{i}_150.pkl', 'rb') as f2:
                            images[i, :, :, :, :] = pickle.load(f2)
                elif test_model == '150l':
                    data_tsne = torch.FloatTensor(10, 150, 2)
                    images = torch.FloatTensor(10, 150, 1, 28, 28)
                    for i in range(10):
                        with open(f'data/fashion_mnist/test/label{i}test_150.pkl', 'rb') as f1:
                            data_tsne[i, :, :] = pickle.load(f1)
                        with open(f'data/fashion_mnist/test/label{i}test_im_150.pkl', 'rb') as f2:
                            images[i, :, :, :, :] = pickle.load(f2)
                else:
                    with open(f'data/fashion_mnist/test/{size}_tsne_data{dataset_number}.pkl', 'rb') as f1:
                        data_tsne = pickle.load(f1)
                    with open(f'data/fashion_mnist/test/{size}_im_data{dataset_number}.pkl', 'rb') as f2:
                        images = pickle.load(f2)

                self.data = [data_tsne, images]

            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'mnist':
            if mode == 'train':
                with open('data/mnist/train/combine_full.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne[0:num_samples * size].reshape(num_samples, size, 2)
            elif mode == 'test':
                with open(f'data/mnist/test/{size}_tsne_data{dataset_number}.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                with open(f'data/mnist/test/{size}_im_data{dataset_number}.pkl', 'rb') as f2:
                    images = pickle.load(f2)

                self.data = [data_tsne, images]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'CIFAR10':
            if mode == 'train':
                with open('data/cifar-10_pt/cifar10_for_show_train_20000_tsne10_big.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                # self.data = data_tsne[(epoch % 5) * size: (epoch % 5 + 5) * size].reshape(5, size, 2)
                self.data = data_tsne[:10*50].reshape(10, size, 2)
            elif mode == 'test':

                n = 10
                with open(f'data/cifar-10_pt/cifar10_for_show_train_10000_tsne5_big.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)[0:n * 50].reshape(1, n*50, 2)
                with open(f'data/cifar-10_pt/cifar10_for_show_train_20000_images5.pkl', 'rb') as f2:
                    images = pickle.load(f2)[0:n * 50].reshape(1, n*50, 3, 32, 32)
                print(data_tsne.size())

                self.data = [data_tsne, images]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'DBLP':
            if mode == 'train':
                # for cora
                with open('data/DBLP/tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne[0:num_samples * size].reshape(num_samples, size, 2)
            elif mode == 'test':
                with open('data/DBLP/tsne.pkl', 'rb') as f1:
                    data_t = pickle.load(f1)[:500, :].reshape(10, 50, 2)
                self.data = [data_t, data_t]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'cora':
            if mode == 'train':
                # for cora
                with open('data/cora/cora_tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne[0:num_samples * size].reshape(num_samples, size, 2)
            elif mode == 'test':
                with open('data/cora/cora_tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)[1000:int((1000+500)), :].reshape(10, 50, 2)
                with open('data/cora/cora_label.pkl', 'rb') as f2:
                    images = pickle.load(f2)[:50]
                self.data = [data_tsne, images]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'FLT':
            if mode == 'train':
                with open('data/FLT/ori_tsne_full.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne[:29,:,:]
                print(data_tsne.size())
            elif mode == 'test':
                with open(f'data/FLT/mean_of_96.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)[:1, :, :]
                self.data = [data_tsne, data_tsne]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'IN':
            if mode == 'train':
                with open(f'data/IN/combine_full_tsne_mix.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                print(data_tsne.size())
                self.data = data_tsne[:size*10].reshape(10, size, 2)
            elif mode == 'test':
                size = 50
                num_samples = 10

                with open(f'data/IN/combine_full_tsne_mix.pkl', 'rb') as f1:
                    data = pickle.load(f1)[:int(size*num_samples), :].reshape(10, size, 2)
                # data = data[:int(size*num_samples)].reshape(num_samples, size, 2)
                with open(f'data/IN/combine_full_tsne_images.pkl', 'rb') as f2:
                    image = pickle.load(f2)[:int(size*num_samples)].reshape(num_samples, size, 3, 64, 64)

                self.data = [data, image]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'sch':
            if mode == 'train':
                with open('data/sch/sch_t_svd32.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne[:242*17, :].reshape(17, 242, 32)
            elif mode == 'test':
                with open('data/sch/sch_tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                print(data_tsne.size())
                self.data = [data_tsne[:242*17, :].reshape(17, 242, 2), data_tsne[:242*17, :].reshape(17, 242, 2)]
        elif mission == 'cars':
            if mode == 'train':
                with open('data/cars/cars_tsne2_val.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne[:300, :].reshape(6, 50, 2)
            elif mode == 'test':
                with open('data/cars/cars_tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                print(data_tsne.size())
                self.data = [data_tsne[299:, :].reshape(1, 99, 2), data_tsne[299:, :].reshape(1, 99, 2)]

        elif mission == 'dum':
            if mode == 'train':
                with open('data/dummy_data/dumpydata_train_x.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                path = 'data/dummy_data/dumpydata_train_y.pkl'
                with open(path, 'rb') as f1:
                    y = pickle.load(f1).reshape(100, 240)[(epoch % 10):(epoch % 10) + 10, :]
                self.data = torch.tensor(data_tsne).reshape(100, 240, 20)[(epoch % 10):(epoch % 10) + 10, :, :]
                save_dataset(y, 'data/dummy_data/dumpydata_train_y_tem.pkl')
        elif mission == 'demo':
            with open(f'data/linshifanhuiwenjian/received.pkl', 'rb') as f1:
                data = pickle.load(f1)

            data = torch.Tensor(data)  # [:, 1:]
            data_tsne = torch.FloatTensor(1, data.size(0), 2)
            images = torch.FloatTensor(1, data.size(0), 1, 1, 1)
            for i in range(1):
                data_tsne[i, :, :] = data
            self.data = [data_tsne, images]
        else: #TSP
            print('Please input right run mission!')
            assert False
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
