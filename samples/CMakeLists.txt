include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/libs/include")

macro(add_sample sample_name)
  add_executable(${sample_name} "${sample_name}.cpp")
  target_link_libraries(${sample_name} PUBLIC fluidsim)
endmacro(add_sample)

add_sample(random)