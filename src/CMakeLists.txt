set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pedantic -pedantic-errors -Werror -Wextra -Wshadow -Wpointer-arith -Wformat=2 -Weffc++ -Wall")

# Optimize options

set(USE_COMPACT_N_SEARCH OFF)

if (USE_COMPACT_N_SEARCH)
  message(STATUS "try to use external `COMPACT_N_SEARCH`")
  message(STATUS "external `COMPACT_N_SEARCH` currently not supported, use internal library")
  # add_subdirectory("${PROJECT_SOURCE_DIR}/libs/CompactNSearch")
  # target_link_libraries(pbf3d_lib PUBLIC CompactNSearch)
endif (USE_COMPACT_N_SEARCH)

# External Library
find_package(glfw3 3.3 REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/libs/include")

file(GLOB source
  "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB lib_source
  "${PROJECT_SOURCE_DIR}/libs/src/*.c*")
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/main.cpp")

message(STATUS "find source: " ${source})
message(STATUS "find lib_source: " ${lib_source})

add_library(pbf3d_lib ${source} ${lib_source})
target_link_libraries(pbf3d_lib PUBLIC glfw ${CMAKE_DL_LIBS})

add_executable(pbf3d "${PROJECT_SOURCE_DIR}/src/main.cpp")

target_link_libraries(pbf3d PUBLIC pbf3d_lib)
