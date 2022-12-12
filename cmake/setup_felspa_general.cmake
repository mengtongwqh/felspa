# ======================================================
#
# GENERAL INFO FOR FELSPA PACKAGE
#
# ======================================================

# -------------------------
# PROJECT NAME
# -------------------------
set(FELSPA_PROJECT_NAME felspa)
set(FELSPA_PROJECT_DESCRIPTION
  "A finite element library for structural geology modelling.")

# -------------------------
# VERSION NUMBER
# -------------------------
# import version from file
file(STRINGS "${CMAKE_SOURCE_DIR}/VERSION" _version LIMIT_COUNT 1)
set_if_empty(FELSPA_VERSION_STRING "${_version}")

# parse version info
string(REGEX
  REPLACE
  "^([0-9]+)\\..*" "\\1"
  FELSPA_VERSION_MAJOR
  "${FELSPA_VERSION_STRING}")

string(REGEX
  REPLACE
  "^[0-9]+\\.([0-9]+).*" "\\1"
  FELSPA_VERSION_MINOR
  "${FELSPA_VERSION_STRING}")

string(REGEX
  REPLACE
  "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1"
  FELSPA_VERSION_PATCH
  "${FELSPA_VERSION_STRING}")

# setting pure numeric version number
set(FELSPA_VERSION_NUMERIC
  ${FELSPA_VERSION_MAJOR}.${FELSPA_VERSION_MINOR}.${FELSPA_VERSION_PATCH})

# ----------------------------
# SET PATHS
# ----------------------------
# relative directories
set(FELSPA_INCLUDE_REL_DIR include)
set(FELSPA_LIBRARY_REL_DIR lib)
set(FELSPA_BINARY_REL_DIR bin)
set(FELSPA_PROJCONFIG_REL_DIR
  ${FELSPA_LIBRARY_REL_DIR}/cmake/${FELSPA_PROJECT_NAME})

# make relative install path absolute
foreach(_path LIBRARY INCLUDE CMAKE BINARY)
  set(_var "${FELSPA_${_path}_REL_DIR}")
  set(FELSPA_INSTALL_${_path}_DIR "${CMAKE_INSTALL_PREFIX}/${_var}")
endforeach()

# ----------------------------
# SET FILE NAME
# ----------------------------
set(CMAKE_DEBUG_POSTFIX ".dbg")
set(FELSPA_LIBRARY_TARGET ${FELSPA_PROJECT_NAME})
