# This script will set up everything related to the C++ compiler. Mainly the C++
# standard and the compiler flags

# ------------------------
# CPP STANDARD
# ------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ------------------------
# BUILD TYPE
# ------------------------

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE
    "Debug" CACHE STRING
    "Build type is defaulted to Debug. You can also choose Release"
    FORCE)
endif()

if(NOT ${CMAKE_BUILD_TYPE} MATCHES "^(Debug|Release)$")
  message(
    FATAL_ERROR "Detected unknown CMAKE_BUILD_TYPE! Must be Debug or Release")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" FELSPA_BUILD_TYPE)

# --------------------------
# COMPILER FLAGS
# --------------------------
list(APPEND FELSPA_CXX_FLAGS -ansi -pedantic)

# DEBUG TARGETS
list(APPEND FELSPA_CXX_FLAGS_DEBUG ${FELSPA_CXX_FLAGS})
append_flag_if_supported(FELSPA_CXX_FLAGS_DEBUG "-g")
append_flag_if_supported(FELSPA_CXX_FLAGS_DEBUG "-Og")
append_flag_if_supported(FELSPA_CXX_FLAGS_DEBUG "-Wall")
append_flag_if_supported(FELSPA_CXX_FLAGS_DEBUG "-Wextra")

# RELEASE TARGETS
list(APPEND FELSPA_CXX_FLAGS_RELEASE ${FELSPA_CXX_FLAGS})
append_flag_if_supported(FELSPA_CXX_FLAGS_RELEASE "-O3")
append_flag_if_supported(FELSPA_CXX_FLAGS_RELEASE "-funroll-all-loops")
append_flag_if_supported(FELSPA_CXX_FLAGS_RELEASE "-fstrict-aliasing")


# --------------------------
# COMPILER DEFINITIONS
# --------------------------
list(APPEND FELSPA_CXX_DEFINITIONS_DEBUG "DEBUG")
list(APPEND FELSPA_CXX_DEFINITIONS_RELEASE "NDEBUG")


# --------------------------
# LINKER FLAGS
# --------------------------
list(APPEND FELSPA_LINKER_FLAGS "-pthread")
list(APPEND FELSPA_LINKER_FLAGS "-rdynamic")
