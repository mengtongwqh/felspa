# ----------------------------------------
# BUILD SYSTEM FOR FELSPA LIBRARY
# ----------------------------------------

cmake_minimum_required(VERSION 3.1.0)
cmake_policy(SET CMP0054 NEW)

# include all useful cmake macros
include("${CMAKE_SOURCE_DIR}/cmake/setup_cmake_macros.cmake")

# ---------------------------
# SETUP GENERAL PARAMETERS
# ---------------------------
# these are variables that are hidden from user modification
cmake_include_verbose("${CMAKE_SOURCE_DIR}/cmake/setup_felspa_general.cmake")

message("")
message(STATUS "***********************************************************")
message(
  STATUS "# Configuring FELSPA Library Version ${FELSPA_VERSION_STRING} #")
message(STATUS "***********************************************************")
message("")

# --------------------------
# PROJECT NAME
# --------------------------
project(
  ${FELSPA_PROJECT_NAME}
  VERSION ${FELSPA_VERSION_NUMERIC}
  DESCRIPTION ${FELSPA_PROJECT_DESCRIPTION})

# --------------------------
# SETUP COMPILER FLAGS
# --------------------------
cmake_include_verbose("${CMAKE_SOURCE_DIR}/cmake/cxx_compiler_checks.cmake")
cmake_include_verbose("${CMAKE_SOURCE_DIR}/cmake/setup_compiler_flags.cmake")

# --------------------------
# FIND DEPENDENT PACKAGES
# --------------------------
# find the package dependency
cmake_include_verbose("${CMAKE_SOURCE_DIR}/cmake/setup_depend_packages.cmake")

# --------------------------
# SETUP CACHED VARIABLES
# --------------------------
# these cached variables allow user intervention of the build process
cmake_include_verbose("${CMAKE_SOURCE_DIR}/cmake/setup_cached_variables.cmake")

# --------------------------
# ACTUAL BUILDING PROCESS
# --------------------------
message("--------------------------------------")

add_subdirectory(cmake/tools)
message(STATUS "FELSPA CMAKE TOOLS COMPILATION DONE.")

add_subdirectory(include)
message(STATUS "FELSPA HEADER INSTALLATION DONE.")

add_subdirectory(src)
message(STATUS "FELSPA SOURCE FILES CONFIGURED.")

add_subdirectory(cmake/config)
message(STATUS "FELSPA CONFIG FILE WRITTEN")

add_subdirectory(tests)
message(STATUS "FELSPA TEST FILES CONFIGURED")

add_subdirectory(examples)
message(STATUS "FELSPA EXAMPLES LOADED")

message("--------------------------------------")

# --------------------------------------
# LIST OF FLAGS/LIBS USED FOR BUILD
# --------------------------------------
message("")
message("##### Summary of Configurations: #####\n")
message("FELSPA build type: ${CMAKE_BUILD_TYPE}\n")
message("FELSPA compiler flags: ${FELSPA_CXX_FLAGS_${FELSPA_BUILD_TYPE}}\n")
message("FELSPA linked Libraries: \n${FELSPA_LINKED_LIBRARIES}\n")
message("FELSPA include directories: \n${FELSPA_INCLUDE_DIRS}\n")
message("FELSPA linker flags: ${FELSPA_LINKER_FLAGS}\n")
message("FELSPA definitions: ${FELSPA_CXX_DEFINITIONS_${FELSPA_BUILD_TYPE}}")
message("--------------------------------------")
