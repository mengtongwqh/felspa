'''
Author       : WU,Qihang wu.qihang@hotmail.com
Date         : 2022-06-10 16:54:37
LastEditors  : WU,Qihang wu.qihang@hotmail.com
LastEditTime : 2022-12-07 14:39:13
FilePath     : /felspa/python/felspanalyst/paraview_filters.py
Description  : 
'''
import numpy as np
import stgeotk as stg
from vtk.util import numpy_support as ns
from particle import FilterAnalyst

from rayleigh_taylor_analysis import RayleighTaylor
from neoarchean import  NeoarcheanTectonics


def foliation_trace(inputs, output):
    input_data = inputs[0].GetPointData()
    npt = inputs[0].GetNumberOfPoints()
    S_normal= ns.vtk_to_numpy(input_data.GetArray("foliation_normal"))

    S_trace = np.zeros((npt, 3))
    S_trace[:, 0] = -S_normal[:, 1]
    S_trace[:, 1] =  S_normal[:, 0]
    # the third component is left zero

    _, Snormal_dip = stg.cartesian_to_line(S_normal).T
    output.PointData.append(S_trace, "foliation_trace")
    output.PointData.append(90.0 - Snormal_dip, "foliation_dip")

    return output


def rayleigh_taylor_particle_deformation(inputs, output,  **kwargs):
    rt = RayleighTaylor(inputs[0])

    # Arrange the vectors columnwise
    eigvec = rt.analyzer.eigenvectors.transpose((0,2,1))

    plot_lin = kwargs.get("plot_lineation", False)
    plot_foln = kwargs.get("plot_foliation", False)

    # flinn data
    flinn = rt.analyzer["flinn"]

    rt.apply_masks_by_dict(**kwargs)
    if plot_lin:  rt.plot_lineations()
    if plot_foln: rt.plot_foliations()

    lineation_mask = rt.analyzer.data_mask
    output.PointData.append(eigvec, "eigenvectors")
    output.PointData.append(eigvec[:,:,2], "lineation")
    output.PointData.append(eigvec[:,:,0], "foliation_normal")
    output.PointData.append(flinn, "flinn")
    output.PointData.append(lineation_mask.astype(int), "lineation_mask")
    
    return output


def neoarchean_particle_deformation(inputs, output, **kwargs):
    rt = NeoarcheanTectonics(inputs[0])

    # Arrange the vectors columnwise
    eigvec = rt.analyzer.eigenvectors.transpose((0,2,1))

    plot_lin = kwargs.get("plot_lineation", False)
    plot_foln = kwargs.get("plot_foliation", False)

    # flinn data
    flinn = rt.analyzer["flinn"]

    rt.set_masks(**kwargs)
    if plot_lin:  rt.plot_lineations()
    if plot_foln: rt.plot_foliations()

    lineation_mask = rt.analyzer.data_mask
    output.PointData.append(eigvec, "eigenvectors")
    output.PointData.append(eigvec[:,:,2], "lineation")
    output.PointData.append(eigvec[:,:,0], "foliation_normal")
    output.PointData.append(flinn, "flinn")
    output.PointData.append(lineation_mask.astype(int), "lineation_mask")
    output.PointData.append(rt.analyzer["deformation_intensity"],"deformation_intensity")
    
    return rt


def kinematic_vorticity(inputs, output):
    input_data = inputs[0]
    extractor = stg.VTKUnstructuredGridExtractor(input_data)

    # assuming the velocity gradient vector "velo_grad_T"
    L = extractor.get_dataset("Gradients").transpose(0,2,1)
    Wk = stg.kinematic_vorticity(L)
    strain_rate = stg.strain_rate(L)

    # export the kinematic vorticity number
    output.PointData.append(Wk, "Wk")
    output.PointData.append(strain_rate, "strain_rate")



def deformation_analysis(inputs, output):
    analyzer = FilterAnalyst(inputs[0])
    flinn = analyzer["flinn"]
    eigvec = analyzer.eigenvectors.transpose((0,2,1))
    output.PointData.append(eigvec, "eigenvectors")
    output.PointData.append(eigvec[:,:,2], "lineation")
    output.PointData.append(eigvec[:,:,0], "foliation_normal")
    # output.PointData.append(flinn, "flinn")
    return output
