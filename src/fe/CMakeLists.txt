set(sources
  cell_data.cc
  sync_iterators.cc
  )

file(GLOB headers
  ${CMAKE_SOURCE_DIR}/include/FELSPA/fe/*.h)

felspa_add_module(obj_fe OBJECT
  ${headers} ${sources})

