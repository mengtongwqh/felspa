macro(promote_to_cache vars)

  foreach(var ${${vars}})
    set(${var} "${${var}}" CACHE ${ARGN})
  endforeach()

endmacro()
