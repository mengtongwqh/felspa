import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from paraview.vtk.numpy_interface.dataset_adapter import UnstructuredGrid
from util import logger
from collections.abc import Iterable

import stgeotk as stg
from stgeotk.utility import seconds_per_year
from stgeotk.stereomath import pole_to_plane

from particle import FileAnalyst, FilterAnalyst

csv_file_suffix = "TimeStepData"
file_prefix = "RayleighTaylor"


class NeoarcheanTectonics:
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[1], str) and isinstance(args[1], str):
            folder_name, fnumber = args[0], args[1]
            self.folder_path = folder_name
            self.analyzer = FileAnalyst(file_prefix, folder_name)
            self.load_temporal_data()
            self.load_file(fnumber)

        elif len(args) == 1 and isinstance(args[0], UnstructuredGrid):
            logger.info("Loaded unstructured grid from vtk.")
            self.analyzer = FilterAnalyst(args[0])
        elif len(args) == 1 and isinstance(args[0], str):
            self.folder_path = args[0]
            self.analyzer = FileAnalyst(file_prefix, self.folder_path)
            self.load_temporal_data()
        else:
            raise RuntimeError("Unexpected input parameters.")

        self.ref_file_number = -1
        self.ref_ptcl_id = []

    def __getitem__(self, key):
        '''
        Alias the analyzer operator[]
        '''
        return self.analyzer[key]

    def load_file(self, fnumber):
        '''
        Load the file with file number.
        Note that this will invalidate all the previous data masks
        '''
        self.analyzer.clear_dataset_mask()
        if isinstance(fnumber, str):
            m = re.search(r"(\d+|\d+\.\d+)Ma", fnumber)
            if m:
                fnumber = str(
                    np.argmin(
                        np.abs(float(m.group(1)) * seconds_per_year * 1.0e6 -
                               self.time_steps)
                    ) + 1)

        logger.info("The geological time is %f Ma",
                    self.time_steps[int(fnumber)-1]/1e6/seconds_per_year)
        self.analyzer.load_file_number(fnumber)

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

    def plot_lineations(self, **kwargs):
        self.analyzer.plot_lineation(
            self.analyzer["level_set"], "level_set", contour=False, **kwargs)

    # def plot_foliations(self, **kwargs):
    #     self.analyzer.plot_foliation(
    #         self.analyzer["level_set"], "level_set", contour=False, **kwargs)

    def set_masks(self, **kwargs):
        self.apply_mask(self.analyzer["deformation_intensity"] > 10.0)

        foliation_strike = kwargs.get("foliation_strike", [90.0, 150.0])
        foliation_symbol = kwargs.get("foliation_symbol", None)
        # bed_dip = kwargs.get("bed_dip", [75.0, 90.0])
        bounding_box = kwargs.get(
            "bounding_box",
            [[-100000, -40000, -20000], [100000, 40000, -5000]])

        level_set_range = kwargs.get("level_set_range", [0000.0, 5000.0])

        level_set_mask = np.logical_and(
            self.analyzer["level_set"] >= level_set_range[0],
            self.analyzer["level_set"] <= level_set_range[1])
        self.apply_mask(level_set_mask)

        # bed_mask = np.logical_and(
        #     self.analyzer["dip"] >= bed_dip[0],
        #     self.analyzer["dip"] <= bed_dip[1])

        foliation_mask = self.mask_by_foliation_strike(
            foliation_strike[0], foliation_strike[1], foliation_symbol)
        # self.apply_mask(bed_mask, foliation_mask, level_set_mask)
        self.apply_mask(foliation_mask, level_set_mask)

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

    def set_ref_strain_state(self, end_time, threshold_strain):
        self.load_file(end_time)
        end_file_number = int(self.analyzer.file_number)
        self.ref_file_number = end_file_number
        end_ptcl_id = self["id"][self["deformation_intensity"]
                                 > threshold_strain]

        self.load_file(1)
        self.ref_ptcl_id = np.intersect1d(
            end_ptcl_id, self["id"], assume_unique=True).astype(int)
        self.ref_ptcl_id.sort()

        if len(self.ref_ptcl_id) == 0:
            raise RuntimeError("No particles will be analyzed.")

        logger.info(
            f"{len(self.ref_ptcl_id)} high strain particles will be analyzed up to file number: {self.ref_file_number}")

        return end_file_number

    def apply_ref_mask(self, *data_array):
        mask = []
        ret_array = []

        # resolve the ids that are in ref_ptcl_id
        mask = [bool(elmt in self.ref_ptcl_id)
                for idx, elmt in enumerate(self["id"])]
        sorted_mask = self["id"][mask].argsort()

        # sort ascending
        for array in data_array:
            ret_array.append(array[mask][sorted_mask])

        if len(ret_array) == 1:
            return ret_array[0]
        else:
            return tuple(ret_array)

    def run_rotation_analysis(self):
        '''
        Integrate horizontal and vertical rotation
        from beginning to the end and dump the data to csv
        '''
        rota_horz = np.zeros(self.ref_ptcl_id.shape)
        rota_vert = np.zeros(self.ref_ptcl_id.shape)
        for ifile in range(1, self.ref_file_number+1):
            self.load_file(ifile)
            velo_grad = self.apply_ref_mask(self["velocity_gradient"])

            vort1 = velo_grad[:, 2, 1] - velo_grad[:, 1, 2]
            vort2 = velo_grad[:, 0, 2] - velo_grad[:, 2, 0]
            vort3 = velo_grad[:, 1, 0] - velo_grad[:, 0, 1]

            logger.info(
                f"Current geological time is {self.time_steps[ifile] / seconds_per_year / 1e6} Ma.")

            dt = self.time_steps[ifile] - self.time_steps[ifile-1]
            rota_horz += dt * 0.5 * vort3
            rota_vert += dt * 0.5 * \
                np.sqrt(vort1**2 + vort2**2) * np.sign(vort1)

            # export vorticity data into txt files
            df = pd.DataFrame({
                "horizontal_rotation":  rota_horz,
                "vertical_rotation": rota_vert
            })
            csv_file = os.path.join(self.analyzer.export_path,
                                    "rotation_" + str(ifile) + ".csv")
            df.to_csv(csv_file)

        return rota_horz, rota_vert

    def plot_foliation_with_rotation(self, file_no):
        self.load_file(file_no)
        csv_file_name = "rotation_" + str(file_no) + ".csv"
        csv_file_name_with_path = os.path.join(
            self.analyzer.export_path, csv_file_name)

        df = pd.read_csv(csv_file_name_with_path)
        rota = df["vertical_rotation"]

        stereonet = self.analyzer.stereonet

        # dataset = stg.LineData()
        # dataset.load_data(self.apply_ref_mask(
        #     self.analyzer.eigenvectors[:, 2, :]), "lineation",
        #     rota, "vertical rotation")

        dataset = stg.LineData()
        pole = pole_to_plane(np.stack((self["strike"], self["dip"]), axis=1))
        dataset.load_data(
            self.apply_ref_mask(pole),
            "strike and dip",
            np.log(self.apply_ref_mask(self["deformation_intensity"])), "vertical rotation")
        # load foliation normal data
        pole_plot = stg.LinePlot(stereonet, dataset, marker='^', s=2)
        stereonet.append_plot(pole_plot)
        stereonet.generate_plots(False)

    def plot_flinn_diagram(self, file_no, ax):
        self.load_file(file_no)
        l1 = self.apply_ref_mask(nt["lambda1"])
        l2 = self.apply_ref_mask(nt["lambda2"])
        l3 = self.apply_ref_mask(nt["lambda3"])

        y = np.log(l1/l2)
        x = np.log(l2/l3)

        ax.scatter(x, y)

    def plot_rotation_diagram(self, file_no, ax):
        self.load_file(file_no)

        m = np.logical_and(self.apply_ref_mask(self["xyz"][:, 2]) > -25000,
                           self.apply_ref_mask(self["level_set"]) > 0)

        csv_file_name = "rotation_" + str(file_no) + ".csv"
        csv_file_name_with_path = os.path.join(
            self.analyzer.export_path, csv_file_name)

        df = pd.read_csv(csv_file_name_with_path)
        rota_vert, rota_horz = df["vertical_rotation"], df["horizontal_rotation"]

        # caxis = self.apply_ref_mask(nt["xyz"][:, 2])
        caxis = self.apply_ref_mask(nt["level_set"])
        colormap = "RdYlGn"
        scale = self.apply_ref_mask(nt["deformation_intensity"])
        collection = ax.scatter(rota_horz[m], rota_vert[m], s=5,
                                 c=caxis[m], cmap=colormap)
        plt.colorbar(collection, ax=ax)


######################################
#               MAIN                 #
######################################
if __name__ == "__main__":
    ThresholdStrain = 10.0
    folder = "/home/wuqihang/research/ModelExport/FELSPA/SwayzeSinistral/2e20_1e19_0.5cm/"
    end_time = "20.0Ma"

    # pickout those deformation intensity
    # greater than a threshold and get their id
    nt = NeoarcheanTectonics(folder)
    nt.set_ref_strain_state(end_time, ThresholdStrain)
    rota_horz, rota_vert = nt.run_rotation_analysis()

    # fig, ax = plt.subplots()
    # nt.plot_foliation_with_rotation(335)
    # nt.plot_rotation_diagram(335, ax)
    # plt.show()
