import numpy as np
import pandas as pd

from tqdm import tqdm
from calculate_ppss import readSWC

from multiprocessing import Pool, cpu_count

def func(tmp_branch_index, tmp_swc, tmp_den_id):
    try:
        tmp_results = []
        tmp_nodelist = []
        tmp_branch2length = {}
        cur_index = tmp_den_id
        while cur_index not in tmp_branch_index:
            if cur_index == -1:
                break
            tmp_nodelist.append(cur_index)
            cur_index = tmp_swc.loc[cur_index, 'parent']

        if (len(tmp_nodelist) == 0):
            tmp_results.append([tmp_den_id, 0, -1, -10])
        elif (len(tmp_nodelist) == 1) and (tmp_swc.loc[tmp_nodelist[0], 'parent'] == -1):
            tmp_results.append([tmp_den_id, 0, -1, -20])
        else:
            if cur_index == -1:
                tmp_nodelist = tmp_nodelist[:-1]

            tmp_parentlist = list(tmp_swc.loc[tmp_nodelist, 'parent'])

            tmp_delta = (tmp_swc.loc[tmp_nodelist, ['x', 'y', 'z']].values - tmp_swc.loc[
                tmp_parentlist, ['x', 'y', 'z']].values)
            tmp_dis = np.sum(np.sqrt(np.sum(tmp_delta * tmp_delta, axis=1)))

            if tmp_nodelist[-1] in list(tmp_branch2length.keys()):
                tmp_results.append([tmp_den_id, tmp_dis, tmp_branch2length[tmp_nodelist[-1]], tmp_nodelist[-1]])
            else:
                cur_start = tmp_nodelist[-1]
                tmp_list = [tmp_nodelist[-1]]
                while cur_start not in tmp_branch_index:
                    cur_start = tmp_swc.loc[tmp_swc['parent'] == cur_start].index
                    if len(cur_start) == 1:
                        cur_start = cur_start[0]
                        tmp_list.append(cur_start)
                    else:
                        break
                tmp_parentlist = tmp_swc.loc[tmp_list, 'parent']
                tmp_delta = (tmp_swc.loc[tmp_list, ['x', 'y', 'z']].values - tmp_swc.loc[
                    tmp_parentlist, ['x', 'y', 'z']].values)
                tmp_branch2length[tmp_nodelist[-1]] = np.sum(np.sqrt(np.sum(tmp_delta * tmp_delta, axis=1)))

                cur_bif = tmp_nodelist[-1]
                level = 0
                while cur_bif != -1:
                    parent = tmp_swc.loc[cur_bif, 'parent']
                    if parent == -1:
                        break

                    if parent in tmp_branch_index:
                        level += 1
                    cur_bif = parent

                branch_level = level + 1
                tmp_results.append([tmp_den_id, tmp_dis, tmp_branch2length[tmp_nodelist[-1]], branch_level])
    except:
        print(tmp_den_id, 'fail')
        tmp_results = [[tmp_den_id, -1, -1, -30]]
    return tmp_results

def _worker(ppss_data):
    # use complete neuron dendrite morphology data
    # den_path = './data/155k_den_1um/1um/SEU-ALLEN_local_SWC_CCFv3'

    # use example neuron dendrite morphology data
    den_path = './data/dendrite_example'

    try:
        target_cell = ppss_data[-1]
        source_cell = ppss_data[-2]
        coords = ppss_data[:3]

        tmp_den_ = readSWC(den_path + '/' + target_cell)[['type', 'x', 'y', 'z', 'r', 'parent']]

        # branch_level
        tmp_ct = tmp_den_['parent'].value_counts()
        tmp_ct = tmp_ct[tmp_ct == 2]
        tmp_branch_index = list(tmp_ct.index)

        mask = (tmp_den_[['x', 'y', 'z']] == coords).all(axis=1)
        found = tmp_den_[mask].copy()
        exists = mask.any()

        if exists:
            tmp_den = found.copy()
        else:
            return False

        tmp_den['target_cell'] = target_cell
        tmp_den['source_cell'] = source_cell

        tmp_results = []
        for j in tmp_den.index:
            tmp_results.append(func(tmp_branch_index, tmp_den_, j))

        cur_indexlist = [xi[0][0] for xi in tmp_results]
        cur_branch_dis_list = [xi[0][1] for xi in tmp_results]
        cur_branch_length_list = [xi[0][2] for xi in tmp_results]
        cur_branch_id_list = [xi[0][3] for xi in tmp_results]

        tmp_den.loc[cur_indexlist, ['branch_dis']] = cur_branch_dis_list
        tmp_den.loc[cur_indexlist, ['branch_length']] = cur_branch_length_list
        tmp_den.loc[cur_indexlist, ['branch_id']] = cur_branch_id_list

        return tmp_den

    except Exception as e:
        print(f'{ppss_data} fail：{e}')
        return pd.DataFrame(columns=['x', 'y', 'z', 'source_cell', 'target_cell', 'branch_level', 'target_region'])


def get_ppss_table_multi(n_jobs=None):
    df = pd.read_csv("./output/ppss_from_pacs.csv")
    df = df.drop_duplicates(subset=['x', 'y', 'z'])
    df = df[['x', 'y', 'z', 'source_cell', 'target_cell']]

    results = df.values

    n_jobs = n_jobs or cpu_count()
    with Pool(n_jobs) as pool:
        buf = list(tqdm(
            pool.imap(_worker, results),
            total=len(results),
            desc='PPSS'
        ))

    all_n = pd.concat(buf, ignore_index=True) \
             .sort_values(['source_cell', 'target_cell']) \
             .reset_index(drop=True)

    all_n.to_csv('./output/ppss_detail_table.csv', index=True)
    return all_n

def one_cell(args):
    i, tmp_df = args
    counts = (tmp_df.groupby(['type', 'branch_level'])
              .size()
              .unstack(fill_value=0)
              .stack())
    long_df = (counts.reset_index(name='counts')
               .assign(target_cell=i)
               .assign(probability=lambda x: x['counts'] / x['counts'].sum()))
    return long_df

if __name__ == "__main__":
    df = get_ppss_table_multi(n_jobs=50)

    path = "./output/ppss_detail_table.csv"
    df = pd.read_csv(path, index_col=0)
    df = df.drop_duplicates(subset=['x', 'y', 'z'])
    
    df.rename(columns={'branch_dis': 'distance2bif', 'branch_id': 'branch_level'}, inplace=True)
    df = df[['target_cell', 'distance2bif', 'branch_length', 'branch_level']]
    df['range'] = df['distance2bif'] / df['branch_length']
    
    df = df[df['range'] != 1]
    df = df.reset_index(drop=True)
    
    df[['type']] = 'n'
    df.loc[df['range'] >= 0.66, 'type'] = 'end'
    df.loc[df['range'] <= 0.33, 'type'] = 'start'
    df.loc[(df['range'] > 0.33) & (df['range'] < 0.66), 'type'] = 'middle'
    
    swc_list = df['target_cell'].unique()
    
    groups = dict(tuple(df.groupby('target_cell')))
    args_list = [(i, groups[i]) for i in swc_list]
    
    n_jobs = 10
    with Pool(n_jobs) as pool:
        tmp_long_list = list(
            tqdm(pool.imap(one_cell, args_list),
                 total=len(args_list),
                 desc='crosstab'))
    
    long_df = pd.concat(tmp_long_list, ignore_index=True)
    long_df.to_csv('./output/ppss_from_pacs_within_segments_branch_order_summary.csv', index=False)

