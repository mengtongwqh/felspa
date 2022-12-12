set(sources
  level_set_stokes.cc
  )

# set(inst_ins
  # geometry.inst.in
  # velocity_field.inst.in
  # )

file(GLOB headers
  ${CMAKE_SOURCE_DIR}/include/FELSPA/coupled/*.h)

felspa_add_module(obj_coupled OBJECT
  ${headers} ${sources} ${inst_ins})

# explicit_instantiate(obj_level_set "${inst_ins}")
