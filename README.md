# Brain-wide Organization of Post-Synaptic Sites: Three Principles
This repository provides the code to calculating predicted post-synaptic sites (PPSS).

## Dependencies
 * Python 3.9.20
 * scipy==1.13.1
 * tqdm==4.67.1
 * pandas==2.1.3
 * numpy==1.26.2
 * SimpleITK==2.5.3 

# Quick start   
```bash
# Create a new environment
conda create -n ppss python=3.9.20
conda activate ppss

# Install dependencies
conda install scipy tqdm pandas numpy
pip install SimpleITK

# calculate (all/example) ppss and save in ./results/results_pac
python calculate_ppss.py

# calculate (all/example) ppss based on bouton sites and save in ./results/results_pb
python calculate_ppss_bouton.py

# get the map between dendrites and brain regions and save in ./output/mouse_celltype.csv
python get_cell_type.py

# get complete ppss table with branch level and save to ./output/ppss_table_(pac/pb).csv,
# then get brain region and save to ./output/ppss_from_(pacs/boutons).csv
python calculate_branch_level.py

# get detail information of ppss for supplementary figure 2, such as the distance between
# ppss and bifurcation, and save to ./output/ppss_from_pacs_within_segments_branch_order_summary.csv
python calculate_branch_distance.py

```

## Reference
Yina Wei, Yuze Liu, Feng Xiong, Fuhui Long, Hanchuan Peng, Brain-wide Organization of Post-Synaptic Sites: Three Principles, bioRxiv, 2026