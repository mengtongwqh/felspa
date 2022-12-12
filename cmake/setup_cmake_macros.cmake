# include external cmake macros
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# user-defined macros
file(GLOB macros "cmake/macros/*.cmake")
foreach(macro ${macros})
  include(${macro})
  message(STATUS "Included user-defined macro: ${macro}")
endforeach()

