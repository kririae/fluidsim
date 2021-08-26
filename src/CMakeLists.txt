# Copy shader files to build directory
set(shader_files p_vert.glsl p_frag.glsl m_vert.glsl m_frag.glsl)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${shader_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g --expt-extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pedantic -pedantic-errors -Wextra -Wshadow -Wpointer-arith -Wformat=2 -Wall")

# Optimize options
set(USE_COMPACT_N_SEARCH OFF)
if (USE_COMPACT_N_SEARCH)
  message(STATUS "try to use external `COMPACT_N_SEARCH`")
  message(STATUS "external `COMPACT_N_SEARCH` currently not supported, use internal library")
  # add_subdirectory("${PROJECT_SOURCE_DIR}/libs/CompactNSearch")
  # target_link_libraries(pbf3d_lib PUBLIC CompactNSearch)
endif (USE_COMPACT_N_SEARCH)

# External Library
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/libs/include")

file(GLOB source
  "${PROJECT_SOURCE_DIR}/src/*.c*")
file(GLOB lib_source
  "${PROJECT_SOURCE_DIR}/libs/src/*.c*")
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/main.cpp")

message(STATUS "find source: " ${source})
message(STATUS "find lib_source: " ${lib_source})

add_library(pbf3d_lib ${source} ${lib_source})
set_target_properties(pbf3d_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(pbf3d_lib PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(pbf3d_lib PUBLIC imgui::imgui glfw ${CMAKE_DL_LIBS} curand)

add_executable(pbf3d "${PROJECT_SOURCE_DIR}/src/main.cpp")

target_link_libraries(pbf3d PUBLIC pbf3d_lib)
