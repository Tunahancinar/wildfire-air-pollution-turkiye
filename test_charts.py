"""Implementation of the wildfire gas analysis module."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

def charts(table, root, year=None):
    if year is not None:
        table = table[table['Year'] == int(year)].copy()
    target = Path(root) / 'test_charts' / (str(year) if year is not None else 'all')
    target.mkdir(parents=True, exist_ok=True)
    if table.empty:
        return []
    names = sorted(table['Fire'].unique())
    gases = ['CO', 'NO2', 'CH2O', 'SO2', 'O3']
    lookup = table.set_index(['Fire', 'Gas'])
    states = ['Not estimable', 'No detection', 'Low', 'Medium', 'High']
    colors = ['#e2e8f0', '#f8fafc', '#f8d9a0', '#8dd3c7', '#428b70']
    matrix = np.array([[states.index(lookup.loc[(name, gas), 'Confidence class']) for gas in gases] for name in names])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    outputs = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    for kind in ['start', 'confidence', 'mann_whitney']:
        fig, ax = plt.subplots(figsize=(16, max(9, len(names) * 0.8 + 2)))
        fig.patch.set_facecolor('white')
        if kind == 'mann_whitney':
            pvals = np.array([[lookup.loc[(name, gas), 'Mann–Whitney p'] for gas in gases] for name in names], dtype=float)
            values = np.where(np.isnan(pvals), 0, np.where(pvals < 0.05, 2, 1))
            ax.imshow(values, cmap=ListedColormap(['#e2e8f0', '#edf2f7', '#87caba']), vmin=0, vmax=2, aspect='auto')
            title = f"Mann–Whitney U | Before vs. during fire — {year or 'all years'}"
            subtitle = 'Two-sided test · unadjusted p < 0.05 highlighted · * fewer than 3 days in either period'
        else:
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
            title = f"CUSUM & PELT | Change onset — {year or 'all years'}" if kind == 'start' else f"Method agreement | Confidence — {year or 'all years'}"
            subtitle = 'C: CUSUM · P: PELT · U: consensus · dates DD.MM.YYYY · † satellite column series' if kind == 'start' else 'Methods agreeing within ±1 day: 3 = High, 2 = Medium, 1 = Low · 0 = No detection'
        for i, name in enumerate(names):
            for j, gas in enumerate(gases):
                row = lookup.loc[name, gas]
                confidence = row['Confidence class']
                if kind == 'start':

                    def date(column):
                        value = row[column]
                        return 'Not estimable' if pd.isna(value) else str(value).replace('Not detected', 'Not detected')
                    label = '\n'.join([f"C: {date('CUSUM onset')}", f"P: {date('PELT onset')}", f"U: {date('Consensus onset')}"])
                elif kind == 'confidence':
                    label = {'Not estimable': 'Not estimable', 'No detection': 'No detection', 'Low': 'LOW', 'Medium': 'MEDIUM', 'High': 'HIGH'}[confidence] + f"\n{int(row['Agreeing method count'])}/3 methods"
                else:
                    p = row['Mann–Whitney p']
                    if pd.notna(p):
                        mark = '*' if min(row['Before n'], row['During n']) < 3 else ''
                        label = f"p = {p:.4g}{mark}\nU = {row['Mann–Whitney U']:g}\nn = {int(row['Before n'])} / {int(row['During n'])}"
                    else:
                        label = f"Not estimable\nn = {int(row['Before n'])} / {int(row['During n'])}"
                if row['Unit'] == 'mol/m2':
                    label += ' †'
                ax.text(j, i, label, ha='center', va='center', fontsize=10, color='white' if kind != 'mann_whitney' and confidence == 'High' else '#142b3b')
        ax.set_xticks(range(5), ['CO', 'NO2', 'CH2O', 'SO2', 'O3'], fontweight='bold', fontsize=14)
        ax.set_yticks(range(len(names)), names, fontsize=11)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=12)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=3)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.suptitle(title, fontsize=21, fontweight='bold', x=0.04, ha='left', y=0.98, color='#18354b')
        fig.text(0.04, 0.925, subtitle, fontsize=11, color='#475569')
        if kind != 'mann_whitney':
            fig.legend(handles=[Patch(facecolor=c, label=s) for c, s in zip(colors, ['Not estimable', 'No detection', 'Low', 'Medium', 'High'])], loc='lower center', ncol=5, bbox_to_anchor=(0.58, 0.02), frameon=False)
            foot = 'Confidence describes method agreement, not probability or causality. † Unconverted satellite column series; no surface concentration inferred.'
        else:
            foot = 'n = before / during valid days. * Small sample: interpret cautiously. Daily independence unverified; no multiplicity correction. † Satellite column series.'
        fig.text(0.04, 0.01, foot, fontsize=10, color='#475569')
        fig.subplots_adjust(left=0.22, right=0.98, top=0.85, bottom=0.09)
        path = target / f'{kind}.png'
        fig.savefig(path, dpi=200, facecolor='white')
        fig.savefig(target / f'{kind}.pdf', facecolor='white')
        plt.close(fig)
        outputs.append(path)
    return outputs
if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'output')
    for path in charts(pd.read_csv(root / 'ALL_FIRES_DETECTION_AND_MANN_WHITNEY.csv'), root):
        print(path)
