
add_executable(explicit_instantiation explicit_instantiation.cc)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/template_arguments.in
  ${CMAKE_BINARY_DIR}/${FELSPA_BINARY_REL_DIR}/template_arguments)

set_target_properties(explicit_instantiation
  PROPERTIES
  LINKER_LANGUAGE "CXX"
  OUTPUT_NAME explicit_inst
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${FELSPA_BINARY_REL_DIR})

target_compile_options(explicit_instantiation PRIVATE
  ${FELSPA_CXX_FLAGS_RELEASE})
