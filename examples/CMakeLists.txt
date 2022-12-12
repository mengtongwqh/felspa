set(examples porphyroclast porphyroclast_Bons1997)

# loop through each src file and add as executables
foreach(example ${examples})
  set(target_name example_${example})

  add_executable(${target_name} ${target_name}.cc)

  add_dependencies(${target_name} ${FELSPA_LIBRARY_TARGET})

  target_include_directories(
    ${target_name} PUBLIC ${CMAKE_SOURCE_DIR}/include/
                          ${CMAKE_BINARY_DIR}/include/ ${FELSPA_INCLUDE_DIRS})

  target_link_libraries(${target_name} PUBLIC ${FELSPA_LINKED_LIBRARIES}
                        ${FELSPA_LIBRARY_TARGET})

  target_compile_options(${target_name}
                         PRIVATE ${FELSPA_CXX_FLAGS_${FELSPA_BUILD_TYPE}})

  target_link_options(${target_name} PUBLIC ${FELSPA_LINKER_FLAGS})

  target_compile_definitions(
    ${target_name}
    PRIVATE $<$<CONFIG:Debug>:${FELSPA_CXX_DEFINITIONS_DEBUG}>
            $<$<CONFIG:Release>:${FELSPA_CXX_DEFINITIONS_RELEASE}>)

endforeach()
