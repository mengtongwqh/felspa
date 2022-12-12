# This script gathers everything needed to build all directories to be included
# for compiling FELSPA
include_directories(${CMAKE_BINARY_DIR}/include/ ${CMAKE_SOURCE_DIR}/include/
  SYSTEM ${FELSPA_INCLUDE_DIRS})

add_library(${FELSPA_LIBRARY_TARGET} SHARED)

# all modules of source files
add_subdirectory(base)
add_subdirectory(coupled)
add_subdirectory(fe)
add_subdirectory(level_set)
add_subdirectory(linear_algebra)
add_subdirectory(mesh)
add_subdirectory(pde)
add_subdirectory(physics)

# gather all dependencies into libfelspa target
# get_property(objects GLOBAL PROPERTY FELSPA_OBJECTS_${FELSPA_BUILD_TYPE})

set_target_properties(
  ${FELSPA_LIBRARY_TARGET}
  PROPERTIES
  DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
  VERSION "${FELSPA_VERSION_NUMERIC}"
  SOVERSION "${FELSPA_VERSION_NUMERIC}"
  LINKER_LANGUAGE "CXX"
  ARCHIVE_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/${FELSPA_LIBRARY_REL_DIR}"
  LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/${FELSPA_LIBRARY_REL_DIR}")

# install library and configs
install(
  TARGETS ${FELSPA_LIBRARY_TARGET}
  COMPONENT library
  EXPORT ${FELSPA_PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${FELSPA_LIBRARY_REL_DIR}
  LIBRARY DESTINATION ${FELSPA_LIBRARY_REL_DIR})

install(
  EXPORT ${FELSPA_PROJECT_NAME}Targets
  DESTINATION ${FELSPA_PROJCONFIG_REL_DIR}
  COMPONENT library)
