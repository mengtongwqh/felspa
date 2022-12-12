# - Find LibBacktrace (backtrace.h, libbacktrace.a, and libbacktrace.so)
# This module defines
#  LibBacktrace_INCLUDE_DIR, directory containing headers
#  LibBacktrace_SHARED_LIB, path to libbacktrace's shared library
#  LibBacktrace_STATIC_LIB, path to libbacktrace's static library
#  LibBacktrace_FOUND, whether libbacktrace has been found

find_path(LibBacktrace_INCLUDE_DIR
  backtrace.h
  HINTS
  ${LibBacktrace_DIR}/include/
  $ENV{LibBacktrace_DIR}/include/
  $ENV{LIBBACKTRACE_DIR}/include/)

if(APPLE)
  find_library(LibBacktrace_SHARED_LIB
    libbacktrace.dylib
    HINTS
    ${LibBacktrace_DIR}/lib/
    $ENV{LibBacktrace_DIR}/include/
    $ENV{LIBBACKTRACE_DIR}/include/)
else()
  find_library(LibBacktrace_SHARED_LIB
    libbacktrace.so
    HINTS
    ${LibBacktrace_DIR}/lib/
    $ENV{LibBacktrace_DIR}/lib/
    $ENV{LIBBACKTRACE_DIR}/lib/)
endif()

find_library(LibBacktrace_STATIC_LIB
  libbacktrace.a
  HINTS
  ${LibBacktrace_DIR}/lib/
  $ENV{LibBacktrace_DIR}/lib/
  $ENV{LIBBACKTRACE_DIR}/lib/)

message(STATUS 
  "libbacktrace header is found at ${LibBacktrace_INCLUDE_DIR}/backtrace.h")
message(STATUS "libbacktrace static library is found at ${LibBacktrace_STATIC_LIB}")
message(STATUS "libbacktrace shared library is found at ${LibBacktrace_SHARED_LIB}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  LibBacktrace
  REQUIRED_VARS
  LibBacktrace_SHARED_LIB
  LibBacktrace_STATIC_LIB
  LibBacktrace_INCLUDE_DIR)
