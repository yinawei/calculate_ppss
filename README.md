# Brain-wide Organization of Post-Synaptic Sites
This repository provides the code to calculating predicted post-synaptic sites (PPSS).

# Quick start   
```bash
conda create -n ppss python=3.9 pip -y
conda activate ppss

python -m pip install \
  numpy==2.0.1 \
  pandas==2.3.3 \
  scipy==1.13.1 \
  tqdm==4.67.1 \
  SimpleITK==2.5.3

# (optional) get example neuron morphology data and save in ./data/(axon/axon_bouton/dendrite)_example
python get_example.py

# calculate (all/example) ppss and save in ./results/results_pac
# Note: (the preloaded kdtree of dendrites are saved in ./cache)
python calculate_ppss.py

# calculate (all/example) ppss based on bouton sites and save in ./results/results_pb
python calculate_ppss_bouton.py

# get the map between dendrites and brain regions and save in ./output/mouse_celltype.csv
python get_cell_type.py

# get complete ppss table with branch level and distance and save to ./output/ppss_table_(pac/pb).csv,
# then get brain region and save to ./output/ppss_from_(pacs/boutons).csv
python calculate_branch_level_with_distance.py

# get detail information of ppss for supplementary figure 2, such as the distance between
# ppss and bifurcation, and save to ./output/ppss_from_pacs_within_segments_branch_order_summary.csv
python calculate_branch_distance.py

```

## Reference
Yina Wei, Yuze Liu, Feng Xiong, Fuhui Long, Hanchuan Peng, Brain-wide Organization of Post-Synaptic Sites: Three Principles, bioRxiv, 2026
