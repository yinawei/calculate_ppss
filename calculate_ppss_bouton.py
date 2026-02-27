import json
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import gc
from scipy.spatial import KDTree
import shutil


def readSWC(swc_path, mode='simple'):
    n_skip = 0
    with open(swc_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break
    names = ["x", "y", "z"]
    used_cols = [0, 1, 2]
    if mode == 'simple':
        pass
    df = pd.read_csv(swc_path, skiprows=n_skip+1, sep=" ",
                     usecols=used_cols,
                     names=names
                     )

    return df


def read_den(swc_path, mode='simple'):
    n_skip = 0
    with open(swc_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break

    names = ["##n", "type", "x", "y", "z", "r", "parent"]
    used_cols = [0, 1, 2, 3, 4, 5, 6]
    if mode == 'simple':
        pass
    df = pd.read_csv(swc_path, index_col=0, skiprows=n_skip, sep=" ",
                     usecols=used_cols,
                     names=names
                     )

    return df


def func(swc1, swc2, swc_id1, swc_id2, dest):
    axon1 = swc1[['x', 'y', 'z']]
    axon1 = axon1.astype(float)
    den2 = swc2.loc[swc2.type.isin([3, 4]), ['x', 'y', 'z']]  # swc2.loc[swc2.type != 2, ['x', 'y', 'z']]  #
    den2 = den2.astype(float)

    del swc1, swc2
    den_tree = KDTree(den2.values)
    dd, ii = den_tree.query(axon1.values, p=2, distance_upper_bound=5)
    del den_tree
    sele_k = np.argwhere(dd <= 5)
    if len(sele_k) > 0:
        ii = ii[sele_k].squeeze()
        pd.DataFrame({
            'axon_id': axon1.iloc[sele_k.reshape(-1).tolist(), :].index,
            'den_id': den2.iloc[ii.reshape(-1).tolist()].index,
            'dis': dd[sele_k.reshape(-1).tolist()]}
        ).to_csv(dest + '/' + swc_id1 + '_' + swc_id2 + '.csv', sep=',')
    return


def connectivity_v2(swc1, swc2, swc_id1, swc_id2, dest):
    try:
        func(swc1, swc2, swc_id1, swc_id2, dest)
        gc.collect()
        return

    except Exception as e:
        print('fail at: ', e, ' : swc: ', swc_id1, ' ', swc_id2)


if __name__ == "__main__":
    # use complete neuron axon morphology data with bouton coordinates
    # axon_src = ['./data/predicted_bouton_locations/bouton_type5']

    # use example neuron axon morphology data with bouton coordinates
    axon_src = ['./data/axon_bouton_example']

    dest = './results/results_pb'

    swc_path_list1 = []
    for src in axon_src:
        swc_list1 = os.listdir(src)
        swc_path_list1.extend([src + '/' + i for i in swc_list1])

    swc_path_list1 = sorted(swc_path_list1)

    # use complete neuron dendrite morphology data
    # src2 = './data/155k_den_1um/1um/SEU-ALLEN_local_SWC_CCFv3'

    # use example neuron dendrite morphology data
    src2 = './data/dendrite_example'

    swc_list2 = os.listdir(src2)
    swc_path_list2 = [src2 + '/' + i for i in swc_list2]
    swc_path_list2 = sorted(swc_path_list2)

    log_path = './log/log_pb'
    json_path = "./data/connection.json"

    cores = 50  # int(multiprocessing.cpu_count() * 0.8)  # multiprocessing.cpu_count()
    print("cores: ", cores)

    os.makedirs(dest, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        connection_dict = json.load(f)

    for path_swc_i in tqdm(swc_path_list1):
        swc_i = path_swc_i.split('/')[-1]

        if os.path.exists(os.path.join(log_path, swc_i)):
            continue

        if swc_i.split('.')[0] in connection_dict.keys():
            dfi = readSWC(swc_path=path_swc_i)

            pool = multiprocessing.Pool(processes=cores)
            for swc_j in connection_dict[swc_i.split('.')[0]]:
                path_swc_j = src2 + '/' + swc_j + '.swc'

                if os.path.exists(dest + '/' + swc_i + '_' + swc_j + '.csv'):
                    continue

                dfj = read_den(path_swc_j)
                (pool.apply_async(connectivity_v2, (dfi, dfj, swc_i, swc_j, dest,)))
                del dfj

            del dfi
            gc.collect()
            pool.close()
            pool.join()
        else:
            continue

        os.makedirs(os.path.join(log_path, swc_i), exist_ok=True)
