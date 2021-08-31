set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g --expt-relaxed-constexpr --expt-extended-lambda -ccbin=${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pedantic -pedantic-errors -Wextra -Wshadow -Wpointer-arith -Wformat=2 -Wall")

#if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# #see https://gitlab.kitware.com/cmake/cmake/-/issues/22037
#  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/clang++")
#endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif ()

# External Library
list(APPEND CMAKE_MODULE_PATH /usr/lib/cmake/OpenVDB)
find_package(OpenVDB REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/libs/include")

file(GLOB SOURCE
  "${PROJECT_SOURCE_DIR}/src/*.c*")
file(GLOB LIBSOURCE
  "${PROJECT_SOURCE_DIR}/libs/src/*.c*")
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/main.cu")

add_library(fluidsim ${SOURCE} ${LIBSOURCE})
target_compile_options(fluidsim PUBLIC -O3)
set_target_properties(fluidsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(fluidsim PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(fluidsim PUBLIC imgui::imgui glfw ${CMAKE_DL_LIBS})
target_link_libraries(fluidsim PUBLIC OpenVDB::openvdb)

add_executable(pbf3d "${PROJECT_SOURCE_DIR}/src/main.cu")
target_compile_options(pbf3d PUBLIC -O3)
target_link_libraries(pbf3d PUBLIC fluidsim)

# Copy shader files to build directory
set(shader_files p_vert.glsl p_frag.glsl m_vert.glsl m_frag.glsl)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${shader_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
