include("${PROJECT_SOURCE_DIR}/etc/google_test.cmake")

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/libs/include")

macro(add_test_exe test_name)
  add_executable("${test_name}" "${test_name}.cu")
  set_target_properties(pbf3d_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries("${test_name}" PUBLIC pbf3d_lib gtest_main gtest)
  add_test(${test_name} ${test_name})
endmacro(add_test_exe)

add_test_exe(compact_hash_correctness)
add_test_exe(pbd_solver_correctness)