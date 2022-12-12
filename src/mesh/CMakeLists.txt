set(sources
  mesh_refine.cc
  )

# set(inst_ins
  # )

file(GLOB headers
  ${CMAKE_SOURCE_DIR}/include/FELSPA/mesh/*.h)

felspa_add_module(obj_mesh OBJECT
  ${headers} ${sources} ${inst_ins})

# explicit_instantiate(obj_mesh "${inst_ins}")
