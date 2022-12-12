'''
Author       : WU,Qihang wu.qihang@hotmail.com
Date         : 2022-11-27 23:30:59
LastEditors  : WU,Qihang wu.qihang@hotmail.com
LastEditTime : 2022-11-30 16:19:47
FilePath     : /felspa/python/felspanalyst/porphyroclast_analysis.py
Description  : Do data analysis for porphyroclast example case
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import math

import stgeotk as stg
from particle import FileAnalyst, FilterAnalyst

file_prefix = "Porphyroclast"


class Porphyroclast:
    def __init__(self, folder_path, fnumber):
        self.folder_path = folder_path
        self.analyzer = FileAnalyst(file_prefix, self.folder_path)
        self.analyzer.load_file_number(fnumber)

    def apply_mask(self, *args):
        self.analyzer.apply_dataset_mask(*args)
        return self.analyzer.data_mask

    def flinn_diagram(self, ax, color_axis=None, **kwargs):
        m = self.analyzer.data_mask
        x = np.log(self.analyzer["l2"][m] / self.analyzer["l1"][m])
        y = np.log(self.analyzer["l3"][m] / self.analyzer["l2"][m])

        flinn_limit = kwargs.pop("flinn_limit", max(np.max(x), np.max(y)))
        # flinn_limit = ((int(100 * flinn_limit) / 5) + 1) / 20.0

        collection = ax.scatter(x, y,
                                c=color_axis[m], cmap='coolwarm', s=10.0,
                                marker='o', linewidths=0.2,
                                edgecolors="black", **kwargs)

        ax.set_aspect("equal", adjustable='box')
        ax.set_xlim([0, flinn_limit])
        ax.set_ylim([0, flinn_limit])
        ax.ticklabel_format(style="sci", scilimits=(-2, 2))

        # ax.set_yticks(np.linspace(0, flinn_limit, 5))
        # ax.set_xticks(np.linspace(0, flinn_limit, 5))

        cb = plt.colorbar(collection, ax=ax, fraction=0.046,
                          pad=0.04, location="bottom")
        # cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2))

        return ax, cb


def plot_flinn_diagrams(pc, ax1, ax2, **kwargs):
    # Unify the flinn limit
    # xmax = np.log(np.max(pc.analyzer["l2"]/pc.analyzer["l1"]))
    # ymax = np.log(np.max(pc.analyzer["l3"]/pc.analyzer["l2"]))
    # xylim = max(xmax, ymax)

    pc.apply_mask(
        pc.analyzer["spawn_time"] == 0.0,
        pc.analyzer["level_set"] > 0.0,
        pc.analyzer["xyz"][:, 1] > -0.05,
        pc.analyzer["xyz"][:, 1] < 0.05)
    # pc.flinn_diagram(ax1, np.linalg.norm(
    #     pc.analyzer["xyz"], axis=1), flinn_limit=xylim, **kwargs)
    # _, cb1 = pc.flinn_diagram(ax1, np.linalg.norm(
    #     pc.analyzer["xyz"], axis=1), **kwargs)
    _, cb1 = pc.flinn_diagram(ax1, np.arctan2(
        pc.analyzer["xyz"][:, 2], pc.analyzer["xyz"][:, 0]) % math.pi/math.pi * 180.0, **kwargs)

    cb1.ax.set_ylabel(r"|x|")

    pc.analyzer.clear_dataset_mask()

    pc.apply_mask(
        pc.analyzer["spawn_time"] == 0.0,
        pc.analyzer["level_set"] <= 0.0,
        pc.analyzer["xyz"][:, 1] > -0.05,
        pc.analyzer["xyz"][:, 1] < 0.05)
    # pc.flinn_diagram(ax2, np.linalg.norm(
    #     pc.analyzer["xyz"], axis=1), flinn_limit=xylim, **kwargs)
    # _, cb2 = pc.flinn_diagram(ax2, np.linalg.norm(
    #     pc.analyzer["xyz"], axis=1), **kwargs)
    _, cb2 = pc.flinn_diagram(ax2, np.arctan2(
        pc.analyzer["xyz"][:, 2], pc.analyzer["xyz"][:, 0]) % math.pi / math.pi * 180.0, **kwargs)
    cb2.ax.set_xlabel(r"|x|")

    return ax1, ax2


if __name__ == "__main__":
    folder = "/home/wuqihang/OneDrive/Research/ModelExport/LevelSetDG/DeformablePorphyroclast/Porphyroclast_100_1_5sec/"
    save_folder = "/home/wuqihang/Dropbox/SharedFolders/Qihang-Shoufa/LevelSetDG/figures/editable/porphyroclast_flinn_diagram/"

    identifier = os.path.basename(os.path.normpath(folder))
    fnumber = 586
    pc = Porphyroclast(folder, fnumber)
    fig, (ax2, ax1) = plt.subplots(1, 2)
    fig.set_size_inches(11, 8.5)
    plot_flinn_diagrams(pc, ax1, ax2)
    save_filename = identifier + '_' + str(fnumber) + ".svg"
    plt.savefig(os.path.join(save_folder, save_filename))
    # plt.show()
