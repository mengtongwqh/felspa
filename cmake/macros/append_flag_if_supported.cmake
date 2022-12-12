# Check if the compiler has the specified _flag add to set_of_flags if the _flag
# exists
macro(append_flag_if_supported set_of_flags flag)

  # getting rid of spaces
  string(STRIP "${flag}" flag_stripped)

  if(flag_stripped)
    # eliminating dash before flags
    string(REGEX REPLACE "^-(.+)$" "\\1" flag_name "${flag_stripped}")
    check_cxx_compiler_flag(${flag_stripped} has_flag_${flag_name})
    # append flags upon successful test
    if(has_flag_${flag_name})
      list(APPEND ${set_of_flags} ${flag})
    endif()
  endif()

endmacro()
