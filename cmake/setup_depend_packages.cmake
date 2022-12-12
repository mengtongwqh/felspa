# This script will find the packages that FELSPA depends upon, set the #define
# flags in the source  and add the relevant compilation/linker flags


message("")
message(STATUS "Setting up dependent library detection ...")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")


# -------------------
# MPI 
# -------------------
if(FELSPA_FORCE_NO_MPI)
  message(STATUS "MPI disabled as per user request.")
  option(FELSPA_HAS_MPI "FELSPA configured with MPI" OFF)
else()
  find_package(MPI)
  if (MPI_CXX_FOUND)
    option(FELSPA_HAS_MPI "FELSPA configured with MPI" ON)
    list(APPEND FELSPA_INCLUDE_DIRS ${MPI_INCLUDE_DIR})
    list(APPEND FELSPA_LINKED_LIBRARIES ${MPI_CXX_LIBRARIES})
    message(STATUS "FELSPA is configured with MPI.")
    message(STATUS "MPI include dir: ${MPI_INCLUDE_PATH}")
    message(STATUS "MPI libraries: ${MPI_CXX_LIBRARIES}")
  else()
    option(FELSPA_HAS_MPI "FELSPA configured with MPI" OFF)
    message(STATUS "MPI NOT Found. FESLPA MPI functionalities will be disabled.")
  endif()
endif()

# -------------------

# -------------------
# DEAL.II
# -------------------
find_package(deal.II 9.3.0 REQUIRED
  HINTS
  ${DEAL_II_DIR}
  ${deal.II_DIR}
  /usr/local/
  $ENV{DEAL_II_DIR})

if(deal.II_FOUND)
  message(STATUS "deal.II library found")
  option(FELSPA_HAS_DEAL_II "configured with deal.II" ON)

  # import all deal.II cmake configs
  deal_ii_initialize_cached_variables()

  # append dealii libs to felspa
  list(APPEND FELSPA_LINKED_LIBRARIES
    "${DEAL_II_LIBRARIES_${FELSPA_BUILD_TYPE}}")
  list(APPEND FELSPA_INCLUDE_DIRS
    "${DEAL_II_INCLUDE_DIRS}")
endif()


# -------------------
# libdl
# -------------------
find_library(DLPATH 
  NAMES libdl.so dl
  HINTS $ENV{DL_DIR})
if(DLPATH)
  option(FELSPA_HAS_DL "configured with libdl" ON)
  list(APPEND FELSPA_LINKED_LIBRARIES ${DLPATH})
  message(STATUS "libdl is found at ${DLPATH}")
else()
  message(WARNING "libdl is not found.")
endif()


# -------------------
# libbacktrace
# -------------------
find_package(LibBacktrace)
if(LibBacktrace_FOUND)
  list(APPEND FELSPA_LINKED_LIBRARIES ${LibBacktrace_STATIC_LIB})
  list(APPEND FELSPA_INCLUDE_DIRS ${LibBacktrace_INCLUDE_DIR})
  option(FELSPA_HAS_LIBBACKTRACE "felspa is configured with backtrace" ON)
else()
  message(WARNING "libbacktrace is not found. \
  This might affect backtrace printing in exceptions handling")
endif()


# -------------------
# ADDR2LINE
# -------------------
find_program(HAS_ADDR2LINE addr2line)
if(HAS_ADDR2LINE)
  option(FELSPA_HAS_ADDR2LINE "found command for addr2line" ON)
endif()


# -------------------
# BOOST
# -------------------
if(APPLE)
  list(APPEND FDAMR_CXX_DEFS_DEBUG "_GNU_SOURCE")
  list(APPEND FDAMR_CXX_DEFS_RELEASE "_GNU_SOURCE")
endif()


if(FELSPA_HAS_LIBBACKTRACE AND FELSPA_HAS_DL)

  # boost_stacktrace_backtrace is header-only
  # so we don't have to search and link to a shared lib
  find_package(Boost)

  if(Boost_FOUND)
    list(APPEND FELSPA_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")
    list(APPEND FELSPA_CXX_DEFINITIONS_DEBUG "BOOST_STACKTRACE_USE_BACKTRACE")
    option(FELSPA_HAS_BOOST_STACKTRACE "configured with boost_stacktrace" ON)
    message(STATUS "Boost stacktrace is configured with libbacktrace")
  endif()

elseif(FELSPA_HAS_ADDR2LINE AND FELSPA_HAS_DL)

  find_package(Boost COMPONENTS stacktrace_addr2line)

  if(Boost_stacktrace_addr2line_FOUND)
    list(APPEND FELSPA_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")
    list(APPEND FELSPA_LINKED_LIBRARIES Boost::stacktrace_addr2line)
    list(APPEND FELSPA_CXX_DEFINITIONS_DEBUG "BOOST_STACKTRACE_USE_ARR2LINE")
    list(APPEND FELSPA_CXX_DEFINITIONS_RELEASE "BOOST_STACKTRACE_USE_ARR2LINE")
    option(FELSPA_HAS_BOOST_STACKTRACE "configured with boost_stacktrace" ON)
    message(STATUS "Boost stacktrace is configured with addr2line program")
  endif()
endif()

if (NOT FELSPA_HAS_BOOST_STACKTRACE)
  # last resort, use basic version
  find_package(Boost COMPONENTS stacktrace_basic)

  if(Boost_stacktrace_basic_FOUND)
    list(APPEND FELSPA_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")
    list(APPEND FELSPA_LINKED_LIBRARIES Boost::stacktrace_basic)
    set(FDAMR_HAS_BOOST_STACKTRACE True)
    message(STATUS "Boost stacktrace is configured with stacktrace_basic.")
  else()
    message(WARNING "Boost stacktrace is not successfully configured.")
  endif()

endif()
    

# -----------------------------

message(STATUS "Library detection completed.")
message("")
