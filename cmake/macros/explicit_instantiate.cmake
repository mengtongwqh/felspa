macro(explicit_instantiate target inst_in_files)

  foreach(inst_in_file ${inst_in_files})

    string(REGEX REPLACE "\\.in$" "" inst_file "${inst_in_file}")
    set(command ${CMAKE_BINARY_DIR}/${FELSPA_BINARY_REL_DIR}/explicit_inst)

    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${inst_file}
      DEPENDS ${CMAKE_BINARY_DIR}/${FELSPA_BINARY_REL_DIR}/template_arguments
      COMMAND
        ${command} ARGS
        ${CMAKE_BINARY_DIR}/${FELSPA_BINARY_REL_DIR}/template_arguments <
        ${CMAKE_CURRENT_SOURCE_DIR}/${inst_in_file} >
        ${CMAKE_CURRENT_BINARY_DIR}/${inst_file}.tmp
      COMMAND
        ${CMAKE_COMMAND} ARGS -E rename
        ${CMAKE_CURRENT_BINARY_DIR}/${inst_file}.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/${inst_file})

    list(APPEND inst_targets ${CMAKE_CURRENT_BINARY_DIR}/${inst_file})

  endforeach()

  # The target for generating instantiation files for this module.
  add_custom_target(${target}_inst ALL DEPENDS ${inst_targets})

  # do not run instantiation if the explicit_instantiation executable has not
  # been built
  add_dependencies(${target}_inst explicit_instantiation)

  # make ${target} depends on the all template inst so that the module will not
  # be compiled before instantiation
  add_dependencies(${target} ${target}_inst)

endmacro()
