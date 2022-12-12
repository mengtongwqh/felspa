set(sources functions.cc viscous_flow.cc material_base.cc)

file(GLOB headers ${CMAKE_SOURCE_DIR}/include/FELSPA/physics/*.h)

felspa_add_module(obj_physics OBJECT ${headers} ${sources} ${inst_ins})
