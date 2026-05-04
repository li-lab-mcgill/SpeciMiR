import os
import pandas as pd
import matplotlib.pyplot as plt
from Global_parameters import PROJ_HOME

data_dir  = os.path.join(PROJ_HOME, "TargetScan_dataset")
df = pd.read_csv(os.path.join(data_dir,  
                              "positive_samples_30_randomized_start.csv"), 
                              sep='\t')

seed_start = df['seed start'].dropna().astype(int)
seed_end = df['seed end'].dropna().astype(int)

# 3. 确定分布范围
min_pos = min(seed_start.min(), seed_end.min())
max_pos = max(seed_start.max(), seed_end.max())

plt.figure(figsize=(8, 6))
plt.hist(seed_start, bins=range(min_pos, max_pos + 2), 
         alpha=0.6, label='seed_start', color="#84c3f7")
plt.hist(seed_end,   bins=range(min_pos, max_pos + 2), 
         alpha=0.6, label='seed_end',
         color='#98a0e6')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Position')
plt.ylabel('Count')
plt.title('Distribution of Seed Start and Seed End')
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(data_dir, 'seed_distribution.png'), 
            dpi=800, bbox_inches='tight')
