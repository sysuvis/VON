from numpy import inf
from torch.utils.data import Dataset
import torch
import pickle
from torchvision.utils import make_grid
from problems.order.state_order import StateOrder
from utils.beam_search import beam_search
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import os
from vector import CGaussKernel, CRBFKernel
import math
from pathlib import Path
from PIL import Image

def create_collage(sorted_1d_filepaths, pixels_per_image=28):
    n_images = len(sorted_1d_filepaths)
    n_images_per_side_x = int(np.ceil(np.sqrt(n_images)))
    n_images_per_side_y = int(np.ceil(n_images / n_images_per_side_x))

    # Create an empty array for the collage
    X = np.zeros((pixels_per_image * n_images_per_side_y, pixels_per_image * n_images_per_side_x, 3))

    for idx, file in enumerate(sorted_1d_filepaths):
        i, j = divmod(idx, n_images_per_side_x)
        im = Image.open(file)
        im.thumbnail((pixels_per_image, pixels_per_image), Image.LANCZOS)
        pixels = np.array(im) / 255

        # Handle grayscale images by converting them to RGB
        if len(pixels.shape) == 2:  # If the image is grayscale
            pixels = np.stack((pixels,) * 3, axis=-1)

        X[i * pixels_per_image:(i + 1) * pixels_per_image, j * pixels_per_image:(j + 1) * pixels_per_image] = pixels

    return X

def plot_grid(*images, figsize=6, fignumber="Filter", titles=None, occurences=False):
    num_plots = len(images)

    plt.close(fignumber)
    print(titles)
    fig = plt.figure(figsize=(figsize * int(min(num_plots, 5)), figsize * int(max(num_plots // 5, 1))), num=fignumber)

    for i, grid in enumerate(images):

        size = grid.shape

        if size[-1] == 1:
            if occurences:
                cmap = None
            else:
                cmap = "gray"
        else:
            cmap = None

        if len(size) == 3:
            ax = fig.add_subplot(((num_plots - 1) // 5) + 1, min(int(num_plots % 5) + (int(num_plots // 5) * 5), 5),
                                 i + 1)
            img = grid.reshape(*size)
            ax.imshow(np.squeeze(img), cmap=cmap, vmin=0)
            ax.set_xticks([])
            ax.set_yticks([])

        if titles is not None:
            ax.set_title(titles[i], fontsize=figsize * 3)

    plt.savefig(f'results_dpq/{titles}.png')
    return 1

#FLAS, DPQ
def distance_preservation_quality(sorted_X, p=2, wrap=False):
    # setup of required variables
    grid_shape = sorted_X.shape[:-1]
    N = np.prod(grid_shape)
    H, W = grid_shape
    flat_X = sorted_X.reshape((N, -1))

    # compute matrix of euclidean distances in the high dimensional space
    dists_HD = np.sqrt(squared_l2_distance(flat_X, flat_X))

    # sort HD distance matrix rows in acsending order (first value is always 0 zero now)
    sorted_D = np.sort(dists_HD, axis=1)

    # compute the expected value of the HD distance matrix
    mean_D = sorted_D[:, 1:].mean()

    # compute spatial distance matrix for each position on the 2D grid
    dists_spatial = compute_spatial_distances_for_grid(grid_shape, wrap)

    # sort rows of HD distances by the values of spatial distances
    sorted_HD_by_2D = sort_hddists_by_2d_dists(dists_HD, dists_spatial)

    # get delta DP_k values
    delta_DP_k_2D = get_distance_preservation_gain(sorted_HD_by_2D, mean_D)
    delta_DP_k_HD = get_distance_preservation_gain(sorted_D, mean_D)

    # compute p norm of DP_k values
    normed_delta_D_2D_k = np.linalg.norm(delta_DP_k_2D, ord=p)
    normed_delta_D_HD_k = np.linalg.norm(delta_DP_k_HD, ord=p)

    # DPQ(s) is the ratio between the two normed DP_k values
    DPQ = normed_delta_D_2D_k / normed_delta_D_HD_k

    return DPQ

def squared_l2_distance(q, p):
    ps = np.sum(p * p, axis=-1, keepdims=True)
    qs = np.sum(q * q, axis=-1, keepdims=True)
    distance = ps - 2 * np.matmul(p, q.T) + qs.T
    return np.clip(distance, 0, np.inf)

def compute_spatial_distances_for_grid(grid_shape, wrap):
    if wrap:
        return compute_spatial_distances_for_grid_wrapped(grid_shape)
    else:
        return compute_spatial_distances_for_grid_non_wrapped(grid_shape)


def compute_spatial_distances_for_grid_wrapped(grid_shape):
    n_x = grid_shape[0]
    n_y = grid_shape[1]

    wrap1 = [[0, 0], [0, 0], [0, 0], [0, n_y], [0, n_y], [n_x, 0], [n_x, 0], [n_x, n_y]]
    wrap2 = [[0, n_y], [n_x, 0], [n_x, n_y], [0, 0], [n_x, 0], [0, 0], [0, n_y], [0, 0]]

    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between on positions on the grid
    d = squared_l2_distance(mat_flat, mat_flat)
    for i in range(8):
        # look for smaller distances with wrapped coordinates
        d_i = squared_l2_distance(mat_flat + wrap1[i], mat_flat + wrap2[i])
        d = np.minimum(d, d_i)

    return d


def compute_spatial_distances_for_grid_non_wrapped(grid_shape):
    # create 2D position matrix with tuples, i.e. [(0,0), (0,1)...(H-1, W-1)]
    a, b = np.indices(grid_shape)
    mat = np.concatenate([np.expand_dims(a, -1), np.expand_dims(b, -1)], axis=-1)
    mat_flat = mat.reshape((-1, 2))

    # use this 2D matrix to calculate spatial distances between on positions on the grid
    d = squared_l2_distance(mat_flat, mat_flat)
    return d

def sort_hddists_by_2d_dists(hd_dists, ld_dists):
    max_hd_dist = np.max(hd_dists) * 1.0001

    ld_hd_dists = hd_dists / max_hd_dist + ld_dists  # add normed HD dists (0 .. 0.9999) to the 2D int dists
    ld_hd_dists = np.sort(ld_hd_dists)  # then a normal sorting of the rows can be used

    sorted_HD_D = np.fmod(ld_hd_dists, 1) * max_hd_dist

    return sorted_HD_D

def get_distance_preservation_gain(sorted_d_mat, d_mean):
    # range of numbers [1, K], with K = N-1
    nums = np.arange(1, len(sorted_d_mat))

    # compute cumulative sum of neighbor distance values for all rows, shape = (N, K)
    cumsum = np.cumsum(sorted_d_mat[:, 1:], axis=1)

    # compute average of neighbor distance values for all rows, shape = (N, K)
    d_k = (cumsum / nums)

    # compute average of all rows for each k, shape = (K, )
    d_k = d_k.mean(axis=0)

    # compute Distance Preservation Gain and set negative values to 0, shape = (K, )
    d_k = np.clip((d_mean - d_k) / d_mean, 0, np.inf)

    return d_k

def create_vectors_from_thumbnails_add(tensor_images, thumb_size=4):
    n_images = tensor_images.shape[0]
    feature_vectors = np.zeros((n_images, thumb_size * thumb_size))

    for i in range(n_images):
        # 从tensor中提取单张图片
        image_array = tensor_images[i, 0, :, :].numpy() * 255  # 将值还原到0-255范围
        im = Image.fromarray(image_array.astype(np.uint8))  # 将numpy数组转换为PIL图像
        im = im.convert('L')  # 确保是灰度图像

        im.thumbnail((thumb_size, thumb_size), Image.ANTIALIAS)
        pixels = np.array(im) / 255.0

        pixels = pixels.flatten()  # 将2D数组扁平化为1D
        feature_vectors[i] = pixels

    return feature_vectors


def list_all_image_paths(images_path):
    return sorted([str(f) for f in Path(images_path).rglob('*.*') if
                   (("jpeg" in str(f).lower()) | ("jpg" in str(f).lower()) | ("png" in str(f).lower())) & (
                               "._" not in str(f).lower()) & (".txt" not in str(f).lower())])

def DPQ_loss(set_folder):

    # image_files = list_all_image_paths(set_folder)
    vectors = create_vectors_from_thumbnails_add(set_folder.cpu(), thumb_size=4)
    n_images = len(vectors)
    n_images_per_site = int(np.floor(np.sqrt(n_images)))
    n_images_to_use = n_images_per_site ** 2
    X = vectors[:n_images_to_use]
    N = np.prod(X.shape[:-1])
    X = X.reshape((n_images_per_site, n_images_per_site, -1))
    dpq = distance_preservation_quality(X, p=16)
    return 1 - dpq

#KS
def compute_gradient(K, L, PI_0):
    grad = np.dot(L, np.dot(PI_0, K))
    grad = 2 * grad
    return grad

def init_eig(K, L, n_obs):
    # with sorted eigenvectors

    [U_K, V_K] = np.linalg.eig(K)
    [U_L, V_L] = np.linalg.eig(L)
    i_VK = np.argsort(-V_K[:, 0])
    i_VL = np.argsort(-V_L[:, 0])
    PI_0 = np.zeros((n_obs, n_obs))
    PI_0[np.array(i_VL), np.array(i_VK)] = 1
    return PI_0

def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or (len(ext) and ext.lower() in extensions):
                yield joined

def find_rectangle(n, max_ratio=2):
    sides = []
    square = int(math.sqrt(n))
    for w in range(square, max_ratio * square):
        h = int(n / w)
        used = int(w * h)
        leftover = n - used
        sides.append((leftover, (w, h)))
    return sorted(sides)[0][1]

def lab(Image):
    # Convert to CIE L*a*b* (CIELAB)
    WhitePoint = np.array([0.950456, 1, 1.088754])
    Image = xyz(Image)  # Convert to XYZ
    # Convert XYZ to CIE L*a*b*
    X = Image[:, :, 0] / WhitePoint[0]
    Y = Image[:, :, 1] / WhitePoint[1]
    Z = Image[:, :, 2] / WhitePoint[2]
    fX = f(X)
    fY = f(Y)
    fZ = f(Z)
    Image[:, :, 0] = 116 * fY - 16    # L*
    Image[:, :, 1] = 500 * (fX - fY)  # a*
    Image[:, :, 2] = 200 * (fY - fZ)  # b*
    return Image

def xyz(Image):
    # Convert to CIE XYZ
    WhitePoint = np.array([0.950456, 1, 1.088754])
    # Undo gamma correction
    R = invgammacorrection(Image[:, :, 0])
    G = invgammacorrection(Image[:, :, 1])
    B = invgammacorrection(Image[:, :, 2])
    # Convert RGB to XYZ
    T = np.linalg.inv(np.array([[3.240479, -1.53715, -0.498535], [-0.969256, 1.875992, 0.041556], [0.055648, -0.204043, 1.057311]]))
    Image[:, :, 0] = T[0, 0] * R + T[0, 1] * G + T[0, 2] * B  # X
    Image[:, :, 1] = T[1, 0] * R + T[1, 1] * G + T[1, 2] * B  # Y
    Image[:, :, 2] = T[2, 0] * R + T[2, 1] * G + T[2, 2] * B  # Z
    return Image

def invgammacorrection(Rp):
    R = np.real(((Rp + 0.099) / 1.099) ** (1 / 0.45))
    i = R < 0.018
    R[i] = Rp[i] / 4.5138
    return R

def f(Y):
    fY = np.real(Y ** (1.0 / 3))
    i = (Y < 0.008856)
    fY[i] = Y[i] * (841.0 / 108) + (4.0 / 29)
    return fY

def KS_loss(iname, n_obs, indexes):
    n = iname.size(0)
    nx = n
    ny = 1
    imgdata = []
    data = []
    for i in range(iname.shape[0]):
        img = iname[i, :, :, :]
        ndarr = make_grid(img).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        if ndarr.shape[2] == 1:
            ndarr = np.repeat(ndarr, 3, axis=2)
        data.append(ndarr.flatten())

        daim = np.double(ndarr) / 255.0
        daimlab = lab(daim)
        imgdata.append(daimlab.flatten())
    imgdata = np.asarray(imgdata)
    griddata = np.zeros((2, ny * nx))
    griddata[0, ] = np.kron(range(1, ny + 1), np.ones((1, nx)))
    griddata[1, ] = np.tile(range(1, nx + 1), (1, ny))
    X_1 = imgdata
    X_2 = griddata.T
    omegas = 1.0
    dk = CRBFKernel()
    dl = CRBFKernel()
    dK = dk.Dot(X_1, X_1)
    dL = dl.Dot(X_2, X_2)
    omega_K = 1.0 * omegas / np.median(dK.flatten())
    omega_L = 1.0 / np.median(dL.flatten())
    kernel_K = CGaussKernel(omega_K)
    kernel_L = CGaussKernel(omega_L)
    K = kernel_K.Dot(X_1, X_1)
    L = kernel_L.Dot(X_2, X_2)
    H = np.eye(n_obs) - np.ones(n_obs) / n_obs
    obj_funct = np.trace(np.dot(np.dot(np.dot(H, K), H), L))
    return 1 / obj_funct

#isomatch
def get_tensor_average_color(tensor):
    img = tensor
    ndarr = make_grid(img).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    if ndarr.shape[2] == 1:
        ndarr = np.repeat(ndarr, 3, axis=2)
    avg_color = np.mean(ndarr, axis=(0, 1))
    return avg_color

def evaluate_objective_func_internal(distances1, distances2):
    C = np.sum(distances1 * distances2) / np.sum(distances1 ** 2)
    result = np.sqrt(np.sum((C * distances1 - distances2) ** 2)) / np.sqrt(np.sum(distances2 ** 2))
    return result

def find_minimizer_l1(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    assert len(x) == len(y)
    candidates = y / x
    result = np.array([np.sum(np.abs(c * x - y)) for c in candidates])
    C_min = candidates[np.argmin(result)]
    C_min = np.where(C_min == -np.inf, 0, C_min)
    C_min = np.where(C_min == np.inf, 0, C_min)
    return C_min

def generate_regular_grid_coordinates(num_rows, num_cols, from_x=0, to_x=1, from_y=0, to_y=1):
    x_vals = np.linspace(from_x, to_x, num_cols)
    y_vals = np.linspace(from_y, to_y, num_rows)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_coordinates = np.vstack([X.ravel(), Y.ravel()]).T
    return grid_coordinates

def isomatch_loss(tensor, num_images, pi):
    avg_colors = []
    for img in range(num_images):
        avg_colors.append(get_tensor_average_color(tensor[img]))
    avg_colors = np.array(avg_colors)
    d_list = pdist(avg_colors)
    d_matrix = squareform(d_list)
    d_list = squareform(d_matrix)
    grid_coords = generate_regular_grid_coordinates(1, int(d_matrix.shape[0]))
    grid_coords1 = grid_coords[:num_images, :]
    loss = evaluate_objective_func_internal(d_list, pdist(grid_coords1))
    return loss


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


class Order(object):

    NAME = 'order'

    @staticmethod
    def get_costs(dataset, pi, cost_choose='image_baseline'):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset)).cuda()

        if cost_choose == 'stress':

            ret_res = torch.zeros(d.size(0)).cuda()
            for i in range(pi.size(0)):

                W = d[i, :, :].cuda()

                new_order = pi[i]

                for b in range(1):
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
                # ret_res[a] = 1 - moransi(d[a, :, :])
                ret_res[a] = 1 - moransi(d[a, :, :])
            return ret_res, None

        elif cost_choose == 'image_baseline':

            ret_res = torch.zeros(d.size(0)).cuda()
            n1 = 50
            name1 = 'CIFAR10'
            type1 = 'n'
            loss1 = 'KS'

            if name1 == 'CIFAR10':
                pa = [n1, 3, 32, 32]
            elif name1 =='IN':
                pa = [n1, 3, 64, 64]
            else:
                pa = [n1, 1, 28, 28]

            # input all the related image train
            # file_path = 'D:/CODE/VON-master/VON/data_baseline/CIFAR10/CIFAR10_50_dis_mix_o_im.pkl'
            # file_path = 'D:/CODE/VON-master/VON/data_baseline/FM/FM_50_dis_mix_train_im.pkl'
            # file_path = 'data_baseline/mnist/mnist_50_dis_mix_o_im.pkl'
            # file_path = 'D:/CODE/VON-master/VON/data_baseline/IN/IN_50_dis_mix_o_im.pkl'

            #test
            if type1 == 'g':
                file_path = f'D:/CODE/VON-master/VON/data_baseline/{name1}t/{name1}t_{n1}_dis_g_o_im.pkl'
            elif type1 == 'l':
                file_path = f'data_baseline/{name1}t/{name1}t_{n1}_dis_l_im_o_train.pkl'
            elif type1 == 'n':
                file_path = f'data_baseline/{name1}t/{name1}t_{n1}_n_im.pkl'
            elif type1 == 'rl':
                file_path = f'D:/CODE/VON-master/VON/data_baseline/{name1}t/{name1}t_{n1}_dis_rl_im.pkl'
            with open(file_path, 'rb') as file:
                image_data = pickle.load(file)#/255 #for IN

            image_data = image_data[:pi.size(0)].reshape(pi.size(0), n1, pa[1]*pa[2]*pa[3]).cuda()
            d_image = image_data.gather(1, pi.unsqueeze(-1).expand_as(image_data)).cuda().reshape(pi.size(0), pa[0], pa[1], pa[2], pa[3])

            # for group_idx in range(100):
            for group_idx in range(image_data.shape[0]):
                
                # loss_isomatch = isomatch_loss(d_image[group_idx], n1, pi[group_idx])
                # ret_res[group_idx] = loss_isomatch

                #train
                # loss_KS = KS_loss(d_image[group_idx].cpu(), n1, pi[group_idx].cpu())
                # ret_res[group_idx] = loss_KS

                #test
                # loss_KS = KS_loss(d_image[group_idx].cpu(), n1, pi[group_idx].cpu())
                # ret_res[group_idx] = 1 / loss_KS

                #FOR TRAIN
                # loss_DPQ = DPQ_loss(d_image[group_idx])
                # ret_res[group_idx] = loss_DPQ

                #FOR TEST
                loss_DPQ = 1 - DPQ_loss(d_image[group_idx])
                ret_res[group_idx] = loss_DPQ

            #plot image results
            # loss2 = np.round(ret_res[0].cpu().numpy(),decimals=2).astype(str)
            #
            # image_folder = f'D:/CODE/KernelizedSorting-master/data_baseline_t1/{name1}t_{n1}_{type1}/0'
            # image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if
            #                fname.endswith(('.png', '.jpg', '.jpeg'))]
            # reordered_image = [image_paths[i] for i in pi[0]]
            # X = create_collage(reordered_image)
            # torch.set_printoptions(precision=3)
            # print(plot_grid(X, titles=[f"{name1} {n1} {type1} {loss1}={loss2}"]))

            return ret_res, None
        else:
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

    def __init__(self, filename=None, size=50, num_samples=32, offset=0, distribution='None', mode='train', mission='IN', dataset_number=1, epoch=0):
        super(OrderDataset, self).__init__()

        if mission == 'FM':
            if mode == 'train':

                with open(f'data_baseline/FMt/FMt_50_dis_mix_o_tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne
            elif mode == 'test':

                name1 = 'FMt'
                n1 = 50
                type1 = 'n'
                file_path = f'D:/CODE/VON-master/VON/data_baseline/{name1}/{name1}_{type1}_{n1}_tsne.pkl'
                with open(file_path, 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne
                images = self.data
                self.data = [data_tsne, images]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'mnist':
            if mode == 'train':
                with open('data_baseline/mnist/mnist_50_dis_mix_o_tsne.pkl', 'rb') as f1:
                    data_tsne = torch.Tensor(pickle.load(f1))
                self.data = data_tsne
            elif mode == 'test':
                n1 = 50
                name1 = 'mnistt'
                type1 = 'n'
                file_path = f'D:/CODE/VON-master/VON/data_baseline/{name1}/{name1}_{type1}_{n1}_tsne.pkl'
                with open(file_path, 'rb') as f1:
                    data_tsne = torch.Tensor(pickle.load(f1))
                images = data_tsne
                self.data = [data_tsne, images]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'CF':
            if mode == 'train':
                with open('data_baseline/CIFAR10/CIFAR10_50_dis_mix_o_tsne.pkl', 'rb') as f1:
                    data_tsne = torch.Tensor(pickle.load(f1))
                self.data = data_tsne
            elif mode == 'test':
                n1 = 50
                name1 = 'CIFAR10t'
                type1 = 'n'
                file_path = f'D:/CODE/VON-master/VON/data_baseline/{name1}/{name1}_{type1}_{n1}_tsne.pkl'
                with open(file_path, 'rb') as f1:
                    data_tsne = pickle.load(f1)[:]
                images = data_tsne
                self.data = [data_tsne, images]
            else:
                print('Please input right run mode!')
                assert False
        elif mission == 'IN':
            if mode == 'train':
                with open('data_baseline/IN/IN_50_dis_mix_o_tsne.pkl', 'rb') as f1:
                    data_tsne = pickle.load(f1)
                self.data = data_tsne
            elif mode == 'test':
                n1 = 100
                name1 = 'INt'
                type1 = 'g'
                file_path = f'D:/CODE/VON-master/VON/data_baseline/{name1}/{name1}_{type1}_{n1}_tsne_avg.pkl'
                with open(file_path, 'rb') as f1:
                    data_tsne = pickle.load(f1)[:]
                self.data = [data_tsne, data_tsne]
            else:
                print('Please input right run mode!')
                assert False
        else: #TSP
            if filename is not None:
                assert os.path.splitext(filename)[1] == '.pkl'

                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
            else:
                # Sample points randomly in [0, 1] square
                self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
