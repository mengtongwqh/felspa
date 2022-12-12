set(sources geometry.cc level_set.cc material_stack.cc reinit.cc
            velocity_field.cc)

set(inst_ins geometry.inst.in velocity_field.inst.in)

file(GLOB headers ${CMAKE_SOURCE_DIR}/include/FELSPA/level_set/*.h)

felspa_add_module(obj_level_set OBJECT ${headers} ${sources})

explicit_instantiate(obj_level_set "${inst_ins}")
