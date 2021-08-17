set(shader_files vert.glsl frag.glsl)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${shader_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


set(USE_COMPACT_N_SEARCH OFF)

if (USE_COMPACT_N_SEARCH)
  message(STATUS "try to use external `COMPACT_N_SEARCH`")
  message(STATUS "external `COMPACT_N_SEARCH` currently not supported, use internal library")
  # add_subdirectory("${PROJECT_SOURCE_DIR}/libs/CompactNSearch")
  # target_link_libraries(pbf3d_lib PUBLIC CompactNSearch)
endif (USE_COMPACT_N_SEARCH)

# CUDA Optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CXX_FLAGS}") # intended to append OMP
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-extended-lambda -use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pedantic -pedantic-errors -Werror -Wextra -Wshadow -Wpointer-arith -Wformat=2 -Weffc++ -Wall")

# External Library
find_package(glfw3 3.3 REQUIRED)
include_directories(" ${PROJECT_SOURCE_DIR}/libs/include")

file(GLOB source
  "${PROJECT_SOURCE_DIR}/src/*.c*")
file(GLOB lib_source
  "${PROJECT_SOURCE_DIR}/libs/src/*.c*")
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/main.cu")

message(STATUS "find source: " ${source})
message(STATUS " find lib_source: " ${lib_source})

add_library(pbf3d_lib STATIC ${source} ${lib_source})
set_target_properties(pbf3d_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(pbf3d_lib PUBLIC glfw ${CMAKE_DL_LIBS})

add_executable(pbf3d "${PROJECT_SOURCE_DIR}/src/main.cu")

target_link_libraries(pbf3d PUBLIC pbf3d_lib)
