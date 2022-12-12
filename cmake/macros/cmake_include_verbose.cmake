# Whenever a cmake file is included, print a message to screen
macro(cmake_include_verbose file)
  message(STATUS "Including CMAKE file: ${file}")
  include(${file})
endmacro()