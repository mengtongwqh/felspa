macro(felspa_add_module library)
  add_library(${library} ${ARGN})

  set_target_properties(${library}
    PROPERTIES
    POSITION_INDEPENDENT_CODE True
    LINKER_LANGUAGE "CXX")

  target_compile_options(${library}
    PUBLIC
    ${FELSPA_CXX_FLAGS_${FELSPA_BUILD_TYPE}})

  target_link_libraries(${library} ${FELSPA_LINKED_LIBRARIES})

  target_include_directories(${library} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

  target_link_options(${library} PUBLIC ${FELSPA_LINKER_FLAGS})

  target_compile_definitions(
    ${library} PUBLIC $<$<CONFIG:Debug>:${FELSPA_CXX_DEFINITIONS_DEBUG}>
    $<$<CONFIG:Release>:${FELSPA_CXX_DEFINITIONS_RELEASE}>)

  target_sources(${FELSPA_LIBRARY_TARGET} PRIVATE $<TARGET_OBJECTS:${library}>)

  message(
    STATUS
    "Module [${library}] added to the [${FELSPA_LIBRARY_TARGET}] build...")
endmacro()
