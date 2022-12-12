set(tests
    rayleigh_taylor
    stokes
    zalesak_disk)

foreach(test ${tests})
  set(target_name test_${test})

  add_executable(${target_name} ${target_name}.cc)

  add_dependencies(${target_name} ${FELSPA_LIBRARY_TARGET})

  target_include_directories(
    ${target_name} 
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${FELSPA_INCLUDE_DIRS})

  target_compile_options(${target_name}
    PUBLIC ${FELSPA_CXX_FLAGS_${FELSPA_BUILD_TYPE}})

  target_link_libraries(${target_name}
    PUBLIC ${FELSPA_LIBRARY_TARGET}
    PUBLIC ${FELSPA_LINKED_LIBRARIES})

  target_compile_definitions(
    ${target_name}
    PUBLIC $<$<CONFIG:Debug>:${FELSPA_CXX_DEFINITIONS_DEBUG}>
            $<$<CONFIG:Release>:${FELSPA_CXX_DEFINITIONS_RELEASE}>)

  target_link_options(${target_name} PUBLIC ${FELSPA_LINKER_FLAGS})

endforeach()
