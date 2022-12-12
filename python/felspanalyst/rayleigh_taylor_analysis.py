import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import logger
from collections.abc import Iterable


from vtk.numpy_interface.dataset_adapter import UnstructuredGrid
# from paraview.vtk.numpy_interface.dataset_adapter import UnstructuredGrid
import stgeotk as stg
from particle import FileAnalyst, FilterAnalyst


csv_file_suffix = "TimeStepData"
file_prefix = "RTFC"


class RayleighTaylor:
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[1], str) and isinstance(args[1], str):
            folder_name, fnumber = args[0], args[1]
            self.folder_path = folder_name
            self.analyzer = FileAnalyst(file_prefix, folder_name)
            self.load_temporal_data()

            m = re.search(r"(\d+|\d+\.\d+)Ma", fnumber)
            if m:
                fnumber = str(
                    np.argmin(np.abs(float(m.group(1)) - self.time_steps))+1)
            self.analyzer.load_file_number(fnumber)

            logger.info("The geological time is  %f Ma",
                        self.time_steps[int(fnumber)-1])

        elif len(args) == 1 and isinstance(args[0], UnstructuredGrid):
            logger.info("Loaded unstructured grid from vtk.")
            self.analyzer = FilterAnalyst(args[0])

        elif len(args) == 1 and isinstance(args[0], str):
            self.folder_path = args[0]
            self.analyzer = FileAnalyst(file_prefix, self.folder_path)
        else:
            raise RuntimeError("Unexpected input parameters.")

    def load_temporal_data(self):
        csv_path = os.path.join(self.folder_path,
                                file_prefix + csv_file_suffix + ".csv")

        if not os.path.exists(csv_path):
            raise RuntimeError(f"csv file {csv_path} does not exist.")

        try:
            with open(csv_path, 'r', encoding="utf-8") as csvfile:
                csv_data = pd.read_csv(csvfile, header=0)
                self.time_steps = csv_data["time"].to_numpy()
                self.velocity_norm = csv_data["velocity_norm"].to_numpy()
        except IOError as e:
            raise e
        except BaseException as error:
            raise RuntimeError(
                "CSV file was found but not correctly parsed.") from error

    def plot_temporal_data(self, **kwargs):
        x = self.time_steps
        y = self.velocity_norm

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        ax.ticklabel_format(useMathText=True, scilimits=(-2, 5))

        use_geological_time = kwargs.get("use_geological_time", True)
        if use_geological_time:
            x = stg.second_to_myr(x)
            y = stg.meter_per_second_to_cm_per_year(y)
            ax.set_xlabel("$t$, Myr")
            ax.set_ylabel("$|v|$, cm/year")
        else:
            ax.set_xlabel("$t$, seconds")
            ax.set_ylabel("$|v|$, m/s")

        ax.plot(x, y, linewidth=1.0)

        if "xlim" in kwargs:
            ax.set_xlim(kwargs["xlim"])

        # annotate the maximum value
        peak_idx = self.peak_velo_norm_index()
        # ax.annotate(
        #     "$t_\max$ = %.2f Myr" % (max_velo_time),
        #     (max_velo_time, max_velo_norm),
        #     xytext=(1.25 * max_velo_time, max_velo_norm),
        #     textcoords="data",
        #     arrowprops=dict(arrowstyle="->"))

        ax.axvline(x[peak_idx], color='grey', linestyle='--', linewidth=0.75)
        ax.axhline(y[peak_idx], color='grey', linestyle='--', linewidth=0.75)
        ax.text(x[peak_idx], ax.get_ylim()[1] * 1.005,
                "$%.2f$" % (x[peak_idx]), horizontalalignment="center")
        ax.text(ax.get_xlim()[1] * 1.005, y[peak_idx],
                "$%.4e$" % (y[peak_idx]), verticalalignment="center")
        # ax.plot(max_velo_time, max_velo_norm, 'k.')

        # if not isinstance(label_points, Iterable):
        #     label_points = [label_points]
        # for pt in label_points:
        #     x = self.time_steps[pt]
        #     y = self.velocity_norm[pt]

        # ax.plot(x, y, 'k.',
        #         markersize=10, markeredgecolor="black", markerfacecolor="red")

        return fig, ax

    def apply_mask(self, *args):
        self.analyzer.apply_dataset_mask(*args)
        return self.analyzer.data_mask

    def mask_by_foliation_strike(self, low, upp, sign=None, stk_dip=None):
        if low >= upp:
            raise RuntimeError(f"lower limit {low} >= upper limit {upp}")
        if stk_dip is None:
            stk_dip = np.array([0.5 * (low + upp), 90.0])
        elif isinstance(stk_dip, list):
            stk_dip = np.array(stk_dip)
        normal = stk_dip
        normal[0] += 90.0
        normal[1] = 90.0 - normal[1]
        normal_vector = stg.line_to_cartesian(normal)

        mask = self.__foliation_strike_mask(
            low, upp) | self.__foliation_strike_mask(180.0 + low, 180.0 + upp)

        if sign is not None:
            signs = np.dot(self.analyzer["level_set_normal"], normal_vector)
            if sign == '+':
                sign_mask = (signs > 0.0)
            elif sign == '-':
                sign_mask = (signs < 0.0)
            return mask & sign_mask

        return mask

    def __foliation_strike_mask(self, low, upp):
        low, upp = low % 360.0, upp % 360.0
        flnstk = self.analyzer["foliation_strike"]

        if upp < low:
            return (flnstk <= upp) | (flnstk >= low)
        return (flnstk >= low) & (flnstk <= upp)

    def run_auto_analysis(self, **kwargs):
        """
        Automatic analysis of a dataset located in folder_path
        """
        self.load_temporal_data()
        max_velo_time, max_velo_norm, max_velo_index = self.peak_velo_norm()
        print(f"Max velocity norm is {max_velo_norm} attained at "
              f"{max_velo_time} Myrs at {max_velo_index}-th time step")

        vnorm_cor = self.velocity_norm - self.velocity_norm[0]
        max_vnorm_cor = max(vnorm_cor)
        vnorms_cor = [0.90 * max_vnorm_cor,
                      0.75 * max_vnorm_cor,
                      0.50 * max_vnorm_cor,
                      0.25 * max_vnorm_cor]

        file_numbers = []
        for vnorm in vnorms_cor:
            file_numbers.append(self._file_number_past_peak_velo(
                vnorm, max_velo_index, vnorm_cor))

        plot_lineation = kwargs.get("plot_lineation", True)
        if plot_lineation:
            for fnumber in file_numbers:
                geo_time = self.time_steps[int(fnumber)-1]
                logger.info("The geological time is %f Ma", geo_time)

                self.analyzer.load_file_number(fnumber)
                self.analyzer.stereonet.info_text = \
                    f"The geological time is {geo_time} Ma"
                self.plot_lineations(
                    level_set_range=[-1000.0, 0.5], foliation_symbol='-')
                self.analyzer.save_plot(str(fnumber) + "_granitoid.svg")
                self.analyzer.stereonet.clear()

                self.analyzer.load_file_number(fnumber)
                self.analyzer.stereonet.info_text = \
                    f"The geological time is {geo_time} Ma"
                self.plot_lineations(
                    level_set_range=[-0.5, 5000.0], foliation_symbol='-')
                self.analyzer.save_plot(str(fnumber) + "_greenstone.svg")
                self.analyzer.stereonet.clear()

        plot_temporal = kwargs.get("plot_temporal_data", True)
        if plot_temporal:
            fig, _ = self.plot_temporal_data(file_numbers[-2])
            path = os.path.join(self.analyzer.export_path,
                                "velocity_norm_time.svg")
            fig.savefig(path, bbox_inches="tight")

    def plot_lineations(self, **kwargs):
        # self.analyzer.plot_lineation(
        #     self.analyzer["level_set"], "level_set", contour=False, cmap_limits=level_set_range, **kwargs)
        self.analyzer.plot_lineation(
            self.analyzer["level_set"], "level_set", contour=False, **kwargs)

    def plot_foliations(self, **kwargs):
        self.analyzer.plot_foliation(
            self.analyzer["level_set"], "level_set", contour=False, **kwargs)

    def apply_masks_by_dict(self, **kwargs):
        foliation_strike = kwargs.get("foliation_strike", [90.0, 150.0])
        foliation_symbol = kwargs.get("foliation_symbol", None)
        bed_dip = kwargs.get("bed_dip", [75.0, 90.0])
        bounding_box = kwargs.get(
            "bounding_box",
            [[-40000, 00000, -20000], [40000, 40000, -5000]])

        bed_mask = np.logical_and(
            self.analyzer["dip"] >= bed_dip[0],
            self.analyzer["dip"] <= bed_dip[1])

        foliation_mask = self.mask_by_foliation_strike(
            foliation_strike[0], foliation_strike[1], foliation_symbol)
        level_set_range = kwargs.get("level_set_range", [-2000.0, 5000.0])
        level_set_mask = np.logical_and(
            self.analyzer["level_set"] >= level_set_range[0],
            self.analyzer["level_set"] <= level_set_range[1])
        self.apply_mask(bed_mask, foliation_mask, level_set_mask)

        self.analyzer.filter_by_bounding_box(bounding_box[0], bounding_box[1])
        return self.analyzer.data_mask

    def _file_number_past_peak_velo(self, v, max_velo_index, vnorm_cor):
        vnorm = vnorm_cor[max_velo_index:]
        return max_velo_index + np.argmin(np.abs(vnorm - v)) + 1

    def peak_velo_norm_index(self):
        if self.time_steps.size == 0 or self.velocity_norm.size == 0:
            raise RuntimeError("Temporal data is not yet loaded.")
        max_index = np.argmax(self.velocity_norm)
        return max_index

    def flinn_diagram(self, ax, color_axis, **kwargs):
        """
        Plot flinn diagram of the strain ellipsoids
        Color the labels with color_axis
        """
        m = self.analyzer.data_mask
        x = np.log(self.analyzer["lambda1"][m] / self.analyzer["lambda2"][m])
        y = np.log(self.analyzer["lambda2"][m] / self.analyzer["lambda3"][m])
        flinn_limit = kwargs.pop("flinn_limit", max(np.max(x), np.max(y)))
        collection = ax.scatter(x, y,
                                c=color_axis[m], cmap='coolwarm', s=10.0,
                                marker='o', linewidths=0.2, edgecolors="black", **kwargs)
        ax.set_aspect("equal", adjustable='box')
        ax.set_xlim([0, flinn_limit])
        ax.set_ylim([0, flinn_limit])
        cb = plt.colorbar(collection, ax=ax)
        return ax


######################################
#               MAIN                 #
######################################
if __name__ == "__main__":
    # folder = "/home/wuqihang/data/ModelExport/SwayzeSinistral/2e20_1e19_0.5cm/
    folder = "/home/wuqihang/research/ModelExport/FELSPA/RayleighTaylor_100-100_4_0_FC/"
    # folder = "/home/wuqihang/research/ModelExport/FELSPA/SwayzeSinistral/1e20_1e19_0.25cm/"
    fig, ax = plt.subplots()
    time =  "150Ma"
    rt = RayleighTaylor(folder, time)
    rt.apply_mask(np.logical_and(
        rt.analyzer["level_set"] > -0.02, rt.analyzer["level_set"] < 0.02))
    rt.flinn_diagram(ax, rt.analyzer["xyz"][:, 2], vmin=0.0)
    plt.show()
    fig.savefig(os.path.join(rt.folder_path, "felspanalytic", time + ".svg"))


    # rt.load_temporal_data()
    # fig, ax = rt.plot_temporal_data(use_geological_time=False, xlim=[0, 1000])
    # fig.savefig("temporal.svg")
    # rt.run_auto_analysis(plot_lineation=True)
