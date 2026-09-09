import os
import shutil
import random
import json

random.seed(42)

json_path = './data/connection.json'

axon_src_dirs = [
    './data/1um_ION_Hipp_full_SWC_CCFv3/ION_Hipp_full_SWC_CCFv3',
    './data/1um_ION_PFC_full_SWC_CCFv3/ION_PFC_full_SWC_CCFv3',
    './data/SEU_1um'
]

axon_bouton_src_dirs = ['./data/predicted_bouton_locations/bouton_type5']

dend_src_dirs = [
    './data/155k_den_1um/1um/SEU-ALLEN_local_SWC_CCFv3'
]

axon_dst_dir = './data/axon_example'
axon_bouton_dst_dir = './data/axon_bouton_example'
dendrite_dst_dir = './data/dendrite_example'
os.makedirs(axon_dst_dir, exist_ok=True)
os.makedirs(axon_bouton_dst_dir, exist_ok=True)
os.makedirs(dendrite_dst_dir, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

axon_items = random.sample(list(data.items()), 30)

def find_file(filename, search_dirs, extensions=['.swc', '.eswc']):
    base_name = filename
    for ext in extensions:
        if filename.endswith(ext):
            base_name = filename[:-len(ext)]
            break

    for dir_path in search_dirs:
        if not os.path.exists(dir_path):
            continue
        for ext in extensions:
            full_path = os.path.join(dir_path, base_name + ext)
            if os.path.exists(full_path):
                return full_path
    return None

axon_copied = []
axon_bouton_copied = []
dendrite_copied = []
missing_files = []

for axon_id, dendrite_dict in axon_items:
    axon_src = find_file(axon_id, axon_src_dirs)
    if axon_src:
        axon_dst = os.path.join(axon_dst_dir, os.path.basename(axon_src))
        shutil.copy2(axon_src, axon_dst)
        axon_copied.append((axon_id, axon_src, axon_dst))
    else:
        print(f"not find {axon_id}")
        continue

    dendrite_ids = list(dendrite_dict.keys())
    for dendrite_id in dendrite_ids:
        dendrite_src = find_file(dendrite_id, dend_src_dirs)
        if dendrite_src:
            dst_name = f"{os.path.basename(dendrite_src)}"
            dendrite_dst = os.path.join(dendrite_dst_dir, dst_name)
            shutil.copy2(dendrite_src, dendrite_dst)
            dendrite_copied.append((axon_id, dendrite_id, dendrite_src, dendrite_dst))
        else:
            print(f"not find {dendrite_id}")

files = os.listdir(axon_bouton_src_dirs[0])
files_ = random.sample(files, 30)
axon_items = [(i.split('.')[0], data[i.split('.')[0]]) for i in files_]

for axon_id, dendrite_dict in axon_items:
    axon_src = find_file(axon_id, axon_bouton_src_dirs)
    if axon_src:
        axon_dst = os.path.join(axon_bouton_dst_dir, os.path.basename(axon_src))
        shutil.copy2(axon_src, axon_dst)
        axon_copied.append((axon_id, axon_src, axon_dst))
    else:
        print(f"not find {axon_id}")
        continue

    dendrite_ids = list(dendrite_dict.keys())
    for dendrite_id in dendrite_ids:
        dendrite_src = find_file(dendrite_id, dend_src_dirs)
        if dendrite_src:
            dst_name = f"{os.path.basename(dendrite_src)}"
            dendrite_dst = os.path.join(dendrite_dst_dir, dst_name)
            shutil.copy2(dendrite_src, dendrite_dst)
            dendrite_copied.append((axon_id, dendrite_id, dendrite_src, dendrite_dst))
        else:
            print(f"not find {dendrite_id}")
