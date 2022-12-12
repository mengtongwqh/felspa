macro(_test_cxx_parallel_algorithm)

  list(APPEND CMAKE_REQUIRED_FLAGS "-ltbb")

  check_cxx_source_compiles(
    "
    #include <execution>
    #if !defined(__cpp_lib_parallel_algorithm) || !defined(__cpp_lib_execution)
    # error \" c++ compiler does not support parallel algorithm.\"
    #endif
    int main() {return 0;} 
    "
    FELSPA_CXX_PARALLEL_ALGORITHM)

endmacro()

_test_cxx_parallel_algorithm()
