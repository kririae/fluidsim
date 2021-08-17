include("${PROJECT_SOURCE_DIR}/etc/google_test.cmake")

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/libs/include")


macro(add_test_exe test_name)
  add_executable("${test_name}" "${test_name}.cpp")
  target_link_libraries("${test_name}" PUBLIC pbf3d_lib gtest_main)
  add_test(${test_name} ${test_name}) # TODO: group tests
endmacro(add_test_exe)

add_test_exe(compact_hash_correctness)
add_test_exe(pbd_solver_correctness)