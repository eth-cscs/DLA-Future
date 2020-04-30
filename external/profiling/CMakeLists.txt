cmake_minimum_required(VERSION 3.12.4)
project(profiling CXX)

add_library(profiling INTERFACE)

target_include_directories(profiling INTERFACE include)
