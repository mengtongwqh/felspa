import os
import numpy as np
import matplotlib.pyplot as plt

import vtk
from vtk.util.numpy_support import vtk_to_numpy

FLOATING_POINT_TOL = 1.0e-10


def vtk_2d_data(filename, array_name):
    if not os.path.exists(filename):
        raise RuntimeError(f"{filename} does not exist")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    points = data.GetPoints()
    n_pts = points.GetNumberOfPoints()
    x = np.zeros(n_pts)
    y = np.zeros(n_pts)

    for i in range(0, n_pts):
        pt = points.GetPoint(i)
        x[i], y[i] = pt[0], pt[1]

    cells = data.GetCells()
    n_cells = cells.GetNumberOfCells()
    connectivity = np.zeros((2 * n_cells, 3))

    for i in range(0, n_cells):
        connectivity[2*i, 0] = cells.GetData().GetTuple(5*i + 1)[0]
        connectivity[2*i, 1] = cells.GetData().GetTuple(5*i + 2)[0]
        connectivity[2*i, 2] = cells.GetData().GetTuple(5*i + 3)[0]

        connectivity[2*i+1, 0] = cells.GetData().GetTuple(5*i + 3)[0]
        connectivity[2*i+1, 1] = cells.GetData().GetTuple(5*i + 4)[0]
        connectivity[2*i+1, 2] = cells.GetData().GetTuple(5*i + 1)[0]

    level_set_data = data.GetPointData().GetArray(array_name)

    level_set = np.zeros(n_pts)
    for i in range(0, n_pts):
        level_set[i] = level_set_data.GetTuple(i)[0]

    return x, y, connectivity, level_set


def vtk_3d_slice_with_fixed_x(filename, array_name, x_position):
    '''
    Cut a slice of the data at x = position
    and export the data
    Of course we can extend the functionality later
    but for now let's stick with a fixed
    '''

    if not os.path.exists(filename):
        raise RuntimeError(f"{filename} does not exist")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    # cells = data.GetCells()
    points = vtk_to_numpy(data.GetPoints().GetData())

    # n_pts = points.GetNumberOfPoints()
    cell_connect = np.reshape(vtk_to_numpy(
        data.GetCells().GetData()), (-1, 9))[:, 1:]

    # cells with points on the slice plane
    cell_mask = abs(points[cell_connect[:, 1]][:, 0] -
                    x_position) < FLOATING_POINT_TOL
    face_connect = np.vstack((
        # cell_connect[cell_mask][:, (1, 3, 5)], cell_connect[cell_mask][:, (3, 7, 5)]))
        cell_connect[cell_mask][:, (1, 2, 5)], cell_connect[cell_mask][:, (5, 2, 6)]))
    level_set_data = vtk_to_numpy(data.GetPointData().GetArray(array_name))
    return points[:, 1], points[:, 2], face_connect, level_set_data


def plot_3d():
    # dir_name = "/home/wuqihang/backup/Qihang/OneDrive/Research/ModelExport/LevelSetDG/rt_solver_stats/porphyroclast/"

    # array_name = "PorphyroclastPorphyroclastLS"

    dir_name = "/home/wuqihang/backup/Qihang/OneDrive/Research/ModelExport/LevelSetDG/rt_solver_stats/AllTestRuns/3d/"
    array_name = "RTLowerLS"

    mesh_densities = ["16x16x16", "16x16x16", "32x32x32", "32x32x32"]
    viscosities = ["100_1", "100_10", "100_1", "100_10"]
    solution_methods = ["ILU_SCR", "ILU_FC", "AMG_FC"]
    line_styles = [":", "--", "-"]
    line_widths = [2.0, 1.0, 0.5]
    colors = ["red", "blue", "black"]

    axes = [None]*4
    fig, ((axes[0], axes[1]), (axes[2], axes[3])
          ) = plt.subplots(2, 2, figsize=(8, 6))
    fig.tight_layout()

    for ax, mesh_density, viscosity in zip(axes, mesh_densities, viscosities):
        lines = []
        labels = []

        for solution_method, line_style, color, width \
                in zip(solution_methods, line_styles, colors, line_widths):
            fn = mesh_density + '_' + solution_method + '_' + viscosity + ".vtu"
            fpn = os.path.join(dir_name, fn)
            if os.path.exists(fpn):
                x, y, face_connect, level_set_data = vtk_3d_slice_with_fixed_x(
                    fpn, array_name, 0.5)
                contourset = ax.tricontour(x, y, face_connect, level_set_data,
                                           [0.0], linestyles=line_style, linewidths=width, colors=color)
                lines.append(contourset.legend_elements()[0][0])
                labels.append(solution_method)

        ax.legend(lines, solution_methods, prop={"size": 8})
        ax.set_aspect("equal")

    return fig, axes


def plot_2d():
    dir_name = "/home/wuqihang/Dropbox/SharedFolders/Qihang-Shoufa/LevelSetDG/data/rt_solver_stats/AllTestRuns/2d"
    array_name = "RTLowerLS"

    mesh_densities = ["32x32", "32x32", "64x64", "64x64"]
    viscosities = ["100_1", "100_10", "100_1", "100_10"]
    solution_methods = ["ILU_SCR", "ILU_FC", "AMG_FC"]
    line_styles = [":", "--", "-"]
    line_widths = [2.0, 1.0, 0.5]
    colors = ["red", "blue", "black"]

    axes = [None]*4
    fig, ((axes[0], axes[1]), (axes[2], axes[3])
          ) = plt.subplots(2, 2, figsize=(8, 6))
    fig.tight_layout()

    for ax, mesh_density, viscosity in zip(axes, mesh_densities, viscosities):
        lines = []
        labels = []

        for solution_method, line_style, color, width \
                in zip(solution_methods, line_styles, colors, line_widths):
            fn = mesh_density + '_' + solution_method + '_' + viscosity + ".vtu"
            fpn = os.path.join(dir_name, fn)
            if os.path.exists(fpn):
                x, y, face_connect, level_set_data = vtk_2d_data(
                    fpn, array_name)
                contourset = ax.tricontour(x, y, face_connect, level_set_data,
                                           [0.0], linestyles=line_style, linewidths=width, colors=color)
                lines.append(contourset.legend_elements()[0][0])
                labels.append(solution_method)

        ax.legend(lines, solution_methods, prop={"size": 8})
        ax.set_aspect("equal")

    return fig, axes


if __name__ == "__main__":
    # dir_name = "/home/wuqihang/Dropbox/SharedFolders/Qihang-Shoufa/LevelSetDG/data/rt_solver_stats/AllTestRuns/3d"
    # file_name = os.path.join(dir_name, "16x16x16_ILU_SCR_100_1_LoF1.vtu")
    # array_name = "RTSCRRTLowerLS"
    # fn = os.path.join(dir_name, file_name)

    # x_ref, y_ref, conn_ref, level_set_ref = vtk_2d_data(
    #     ref_file, "RayleighTaylorTestLowerLS")

    # grab three different vtks from three different solution

    # fig1, axes = plot_3d()
    # fig1, axes = plot_2d()
    fig2, axes = plot_3d()
    # fig1.savefig("2d.svg")
    fig2.savefig("3d.svg")
    plt.show()

    # graph them

    # compute relative solution difference in l2-norm

    # for icycle in range(0, 5):
    #     fig, ax = plt.subplots()
    #     ax.set(aspect=1)
    #     ax.set_xlim([-15, 15])
    #     ax.set_ylim([10, 40])
    #     ax.tricontour(x_ref, y_ref, conn_ref, level_set_ref,
    #                   [0.0], colors="black", linewidths=1.0)
    # filename = \
    #     "ZalesaksDisk_Cycle" + str(icycle) + "ZalesakDisk_000000003.vtu"
    # file = os.path.join(dirname, filename)
    # x, y, conn, level_set = load_vtk(file)
    # ax.tricontour(x, y, conn, level_set, [
    #               0.0], linewidths=1.0, colors="black")
    # fig.savefig("Cycle" + str(icycle) + ".svg")
    # plt.plot()
