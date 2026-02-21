import os
import re
import sys
from textwrap import wrap
import matplotlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from const import OUT_PATH

###################### 绘制子网络图所需函数 ######################
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    
    return new_cmap


def line_wrap_label(label:str, width:int=13):
    words = label.split(' ')
    lines = []
    current_line = ''

    for word in words:
        # 如果当前行加上新单词不超过宽度，则添加到当前行
        if len(current_line) + len(word) + 1 <= width:
            current_line += (' ' + word if current_line else word)
        else:  # 当前行加上新单词超过宽度
            if '-' in word or ':' in word:
                parts = re.split('([-:])', word)
                for i in range(0, len(parts), 2):
                    part = parts[i]
                    seperator = parts[i + 1] if i + 1 < len(parts) else ''
                    combined = part + seperator
                    if len(current_line) + len(combined) <= width:
                        current_line += combined
                    else:
                        if current_line: lines.append(current_line)
                        current_line = combined
            else:
                # 如果单词不包含连字符，直接换行
                lines.append(current_line)
                current_line = word
    if current_line:
        lines.append(current_line)

    return '\n'.join(lines).strip()


def show_3D_params_plot(metrics):
    q = 'Lambda==0.1 & Target<1 & Disease<1 & GO<1 & GO!=0'
    q = 'Lambda==0.1 & Target>=1 & Disease>=1 & GO>=1 & GO!=0'
    # q = 'Lambda==0.1 & GO!=0'
    # data = metrics[metrics['Lambda']==0.1][['Compound','Target','GO','Disease','D2C_auc','D2H_auc']]
    data = metrics.query(q)[['Compound','Target','GO','Disease','D2C_auc','D2H_auc']]

    fig = plt.figure(figsize=(8,4), dpi=300)
    plt.rcParams.update({"font.size":7})
    ax = plt.axes(projection="3d")
    # ax.tick_params('both',size=5)

    # ax.w_xaxis.gridlines.set_lw(0.5)
    # ax.w_yaxis.gridlines.set_lw(0.5)
    # ax.w_zaxis.gridlines.set_lw(0.5)
    
    # ax.w_xaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 1)}})
    
    # ax.set_xlim(-3,3)
    # ax.set_ylim(-3,3)
    # ax.set_zlim(-3,3)

    sctt = ax.scatter(
        np.log10(data['Target']),
        np.log10(data['GO']),
        np.log10(data['Disease']),
        s=10,
        c=data['D2C_auc'],
        cmap='coolwarm',
        linewidths=0
    )
    
    for i in range(len(data)):
        ax.text(
            np.log10(data['Target'].iloc[i]),
            np.log10(data['GO'].iloc[i]),
            np.log10(data['Disease'].iloc[i]),
            round(data['D2C_auc'].iloc[i],3),
            alpha=0.8,
            size=3
        )
    
    
    ax.set_xlabel(f"x: log10(T/C)")
    ax.set_ylabel(f"y: log10(G/C)")
    ax.set_zlabel(f"z: log10(D/C)")
    
    
    axpos = ax.get_position()
    caxpos = matplotlib.transforms.Bbox.from_extents(
        axpos.x1 + 0.12,
        axpos.y0,
        axpos.x1 + 0.12 + 0.03,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)
    fig.colorbar(sctt, cax = cax, shrink = 0.6, aspect = 5)

    plt.show()


if __name__ == '__main__':
    # metrics = pd.read_csv(OUT_PATH+'ranks_and_metrics/canberra_40_metrics.csv')
    # show_3D_params_plot(metrics)
    # import sys
    # if sys.platform == 'win32':
    #     metrics = pd.read_csv(OUT_PATH+'ranks_and_metrics/canberra_40_metrics.csv')
    #     show_3D_params_plot(metrics)
    # else:
    #     print('wrong platform!')
    
    l = line_wrap_label('kdg:aps')
    print(l)