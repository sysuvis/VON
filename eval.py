import math
import torch
import os
import argparse
from numpy import inf
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')
import seaborn as sns
from sklearn.manifold import TSNE
import problems
from torchvision import datasets, transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
import numpy as np
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.cluster import KMeans
import random
import pickle


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """


    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=None, num_samples=val_size, offset=opts.offset + val_size * i, mode=opts.run_mode, mission=opts.mission, dataset_number=opts.dataset_number)
    #for test
    dataset = dataset[0]
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        datasett = model.problem.make_dataset(filename=None, num_samples=opts.val_size, offset=opts.offset, size=opts.sample_size, mode=opts.run_mode, dataset=opts.dataset, dataset_number=opts.dataset_number)
        #for test
        datasett = datasett[0]

        results, neworder= _eval_dataset(model, datasett, width, softmax_temp, opts, device)

    save_dataset(neworder[0], f'data/linshifanhuiwenjian/1.pkl')
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    print("First cost: {}".format(costs[0]))
    print("Mean cost: {}".format(np.mean(costs)))
    print(costs, neworder)

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))

    return neworder


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    return_batch = []
    return_order = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        return_batch.append(batch)
        batch = move_to(batch, device)

        start = time.time()
        assert width == 0, "Do not set width when using greedy"
        assert opts.eval_batch_size <= opts.max_calc_batch_size, \
            "eval_batch_size should be smaller than calc batch size"
        batch_rep = 1
        iter_rep = 1
        assert batch_rep > 0
        sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
        batch_size = len(costs)
        ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            seq = seq.tolist()


            results.append((cost, seq, duration))
            return_order.append(seq)

    return results, return_order


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    #yu add
    parser.add_argument('--run_mode', default='test', help="train, test")
    parser.add_argument('--dataset', default='fashion_mnist', help="dataset name")
    parser.add_argument('--dataset_number', default=1, help='For test.')
    parser.add_argument('--positional_encoding', default='PEP', help="PEP, PEF, PEB")
    parser.add_argument('--cost_choose', default='tsp', help="stress, moransI, tsp")
    parser.add_argument('--sample_size', default=100, help="20, 50, 100")

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]


    for width in widths:
        # for dataset_path in opts.datasets:
        eval_dataset(None, width, opts.softmax_temperature, opts)
