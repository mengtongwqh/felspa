set(sources
  base_classes.cc
  function.cc
  exceptions.cc
  io.cc
  log.cc
  quadrature.cc
  utilities.cc
  )

file(GLOB headers
  ${CMAKE_SOURCE_DIR}/include/FELSPA/base/*.h)

felspa_add_module(obj_base OBJECT ${headers} ${sources})
