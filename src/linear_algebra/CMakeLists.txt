set(sources
    ilu.cc
    # numpy_export.cc
    linear_system.cc solution_vector.cc system_assembler.cc)

set(inst_ins # numpy_export.inst.in
    linear_system.inst.in system_assembler.inst.in)

file(GLOB headers ${CMAKE_SOURCE_DIR}/include/FELSPA/linear_algebra/*.h)

felspa_add_module(obj_linear_algebra OBJECT ${headers} ${sources} ${inst_ins})
explicit_instantiate(obj_linear_algebra "${inst_ins}")
