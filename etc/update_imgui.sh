# !/bin/bash
# This is a script written to update ImGui from https://github.com/ocornut/imgui
# Files to update:
#   libs/src/imgui_impl_glfw.cpp - https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp
#   libs/src/imgui_impl_opengl3.cpp - https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp
# -- header files
#   libs/include/imgui_impl_glfw.h - https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h
#   libs/include/imgui_impl_opengl3.h - https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h
#   libs/include/imgui_impl_opengl3_loader.h - https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h

cd ../libs
# echo "Switch current dir to" `pwd`
SOURCE_DIR=`pwd`
cd src
wget "https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.cpp" -O "imgui_impl_glfw.cpp"
wget "https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.cpp" -O "imgui_impl_opengl3.cpp"
cd ..include
wget "https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_glfw.h" -O "imgui_impl_glfw.h"
wget "https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3.h" -O "imgui_impl_opengl3.h"
wget "https://raw.githubusercontent.com/ocornut/imgui/master/backends/imgui_impl_opengl3_loader.h" -O "imgui_impl_opengl3_loader.h"