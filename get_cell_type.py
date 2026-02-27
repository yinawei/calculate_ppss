from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import SimpleITK as sitk

anno = sitk.GetArrayFromImage(sitk.ReadImage(
        "./data/annotation_25.nrrd"))

den_features = pd.read_csv('./data/155k_DEN_soma_feature.csv')

def worker(f):
    m = den_features[den_features['swc_id']==f].values[0, 1:4]

    try:
        p = pd.DataFrame(data=[m], columns=['z', 'y', 'x'])
        p['x'] = np.floor(p['x'] / 25.)
        p['y'] = np.floor(p['y'] / 25.)
        p['z'] = np.floor(p['z'] / 25.)

        p = p.round(0).astype(int)
        if ((p.x.iloc[0] >= 0) & (p.x.iloc[0] < anno.shape[0]) &
                (p.y.iloc[0] >= 0) & (p.y.iloc[0] < anno.shape[1]) &
                (p.z.iloc[0] >= 0) & (p.z.iloc[0] < anno.shape[2])
        ):
            id_ = anno[p.x.iloc[0], p.y.iloc[0], p.z.iloc[0]]
        else:
            id_ = 'unknow'

    except IndexError:
        id_ = 'unknow'
    return f, id_

if __name__ == '__main__':
    csv = pd.read_csv('./data/Mouse.csv', usecols=[1, 2])
    
    lutidtoname = {}
    for i in csv.iterrows():
        lutidtoname[i[1][0]] = i[1][1]

    print(anno.shape)
    
    n_workers = 50
    files = list(den_features['swc_id'])

    idxlist, balist = [], []
    temp = []

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for fname, ba in tqdm(exe.map(worker, files), total=len(files)):
            idxlist.append(fname)
            temp.append(ba)

    for ba in temp:
        if ba == 'unknow':
            balist.append('unknow')
        else:
            balist.append(lutidtoname.get(ba, ba))

    pd.DataFrame(np.array([balist]).T,index=idxlist,columns=['CellType']).to_csv('./output/mouse_celltype.csv')
