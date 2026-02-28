import os
import pandas as pd
from tqdm import tqdm
from calculate_ppss import readSWC
from multiprocessing import Pool, cpu_count

import json

def _worker(args):
    # use complete neuron dendrite morphology data
    # den_path = './data/155k_den_1um/1um/SEU-ALLEN_local_SWC_CCFv3'
    
    # use example neuron dendrite morphology data
    den_path = './data/dendrite_example'
    
    file_name, dest = args
    i = file_name

    try:
        tmp_r = pd.read_csv(dest + '/' + i, sep=',', index_col=0)
        tmp_r = tmp_r.drop_duplicates('den_id')
        tmp_den_ = readSWC(den_path + '/' + i.split('.swc')[1][1:-4] + '.swc')
        tmp_den = tmp_den_.loc[tmp_r['den_id'], ['x', 'y', 'z']]
        tmp_den['source_cell'] = i.split('.swc')[0]
        tmp_den['target_cell'] = i.split('.swc')[1][1:-4] + '.swc'

        # branch_level
        tmp_ct = tmp_den_['parent'].value_counts()
        tmp_ct = tmp_ct[tmp_ct == 2]
        tmp_branch_index = list(tmp_ct.index)

        tmp_results = []
        for j in tmp_r.index:
            cur_index = tmp_r.loc[j, 'den_id']
            tmp_n_branch = 1
            while cur_index != -1:
                cur_index = tmp_den_.loc[cur_index, 'parent']
                if cur_index in tmp_branch_index:
                    tmp_n_branch = tmp_n_branch + 1
            tmp_results.append([tmp_r.loc[j, 'den_id'], tmp_n_branch])

        tmp_den['branch_level'] = -1
        i = [xi[0] for xi in tmp_results]
        j = [xi[1] for xi in tmp_results]
        tmp_den.loc[i, 'branch_level'] = j
        return tmp_den

    except Exception as e:
        print(f'{file_name} fail：{e}')
        return pd.DataFrame(columns=['x', 'y', 'z', 'source_cell', 'target_cell', 'branch_level', 'target_region'])


def get_ppss_table_multi(n_jobs=None, dest='./results/results_pac',
                         save_path='./output/ppss_table_pac.csv'):
    results_list = os.listdir(dest)

    n_jobs = n_jobs or cpu_count()
    args_list = [(item, dest) for item in results_list]

    with Pool(n_jobs) as pool:
        buf = list(tqdm(
            pool.imap(_worker, args_list),
            total=len(results_list),
            desc='PPSS'
        ))

    all_n = pd.concat(buf, ignore_index=True) \
             .sort_values(['source_cell', 'target_cell']) \
             .reset_index(drop=True)

    all_n.to_csv(save_path, index=True)
    return all_n

def add_region_info(meta_file='./output/ppss_table_pac.csv',
                    save_path='./output/ppss_from_pacs.csv'):

    meta = pd.read_csv("./output/mouse_celltype.csv")
    corticalRegion2region = './data/corticalRegion2region.json'
    with open(corticalRegion2region) as f:
        corticalRegion2region = json.load(f)

    meta['target_cell'] = meta['Unnamed: 0']
    meta['target_region'] = (meta['CellType'].apply(lambda x: corticalRegion2region.get(x, 'unknow')))

    meta.drop(columns=['Unnamed: 0', 'CellType'], inplace=True)

    df = pd.read_csv(meta_file, index_col=0)
    df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)

    region_map = meta.set_index('target_cell')['target_region'].to_dict()

    df['target_region'] = df['target_cell'].map(region_map)

    df.to_csv(save_path)


if __name__ == "__main__":
    # get ppss based on axon arbors
    get_ppss_table_multi(n_jobs=10, dest='./results/results_pac',
                         save_path='./output/ppss_table_pac.csv')

    add_region_info(meta_file='./output/ppss_table_pac.csv',
                    save_path='./output/ppss_from_pacs.csv')

    # get ppss based on axon bouton sites
    get_ppss_table_multi(n_jobs=10, dest='./results/results_pb',
                         save_path='./output/ppss_table_pb.csv')

    add_region_info(meta_file='./output/ppss_table_pb.csv',
                    save_path='./output/ppss_from_boutons.csv')



