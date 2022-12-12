# only set the variable if it is empty
macro(set_if_empty var)
  if(NOT "${${var}}")
    set(${var} ${ARGN})
  endif()
endmacro()
