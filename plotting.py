import pandas as pd
import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import matplotlib.ticker as mticker

from pathlib import Path

validation_dir = Path('validation')
val_files = sorted(validation_dir.glob('**/val_results_*'))

with open(val_files[3]) as f:
    val = json.loads(f.read())

best_trial_dir = Path(val['checkpoint_dir']).parent
best_trial_name = best_trial_dir.parts[-1]
ray_result_path = best_trial_dir.parent
print(ray_result_path)
trials_sorted = sorted(ray_result_path.glob('DEFAULT*'))
trial_names = [pathobj.parts[-1] for pathobj in trials_sorted]
best_trial_id = trial_names.index(best_trial_name)

log_list = []
trials_to_plot = range(0, len(trials_sorted))
#trials_to_plot = [0, 11]
for i, ray_trial in enumerate(trials_sorted):
    if i in trials_to_plot:
        log = ray_trial/'progress.csv'
        df = pd.read_csv(log)
        df['trial_id'] = i
        log_list.append(df)

fig, ax = plt.subplots()
for df in log_list:
    ap = df['avg_prec'].values
    trial_id = df['trial_id'][0]
    x = np.arange(0, len(ap), 1)
    linewidth = 1
    linestyle = '--'
    marker = None
    label = f'Trial {trial_id}'
    if trial_id == best_trial_id:
        linewidth = 1.5
        linestyle = '-'
        marker = '+'
        label = label + ' (Best Trial)'
    ax.plot(x,
            ap,
            label=label,
            marker=marker,
            linewidth=linewidth,
            linestyle=linestyle)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Average Precision')
ax.legend()
    
pdb.set_trace()

