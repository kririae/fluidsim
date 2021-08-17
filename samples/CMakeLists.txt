include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/libs/include")

macro(add_sample sample_name)
  add_executable(${sample_name} "${sample_name}.cpp")
  target_link_libraries(${sample_name} PUBLIC pbf3d_lib)
endmacro(add_sample)

add_sample(random)
add_sample(random_gravity)