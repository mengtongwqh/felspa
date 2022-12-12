set(sources
    advection.cc
    boundary_conditions.cc
    diffusion.cc
    hamilton_jacobi.cc
    ldg.cc
    linear_systems.cc
    pde_base.cc
    pde_tools.cc
    stokes_common.cc
    stokes_ilu.cc
    stokes_trilinos.cc
    time_integration.cc)

set(inst_ins advection.inst.in boundary_conditions.inst.in
             hamilton_jacobi.inst.in ldg.inst.in time_integration.inst.in)

file(GLOB headers ${CMAKE_SOURCE_DIR}/include/FELSPA/pde/*.h)

felspa_add_module(obj_pde OBJECT ${headers} ${sources} ${inst_ins})
explicit_instantiate(obj_pde "${inst_ins}")
