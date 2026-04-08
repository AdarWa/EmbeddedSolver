cmake_minimum_required(VERSION 3.16)
project(EmbeddedSolverTraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRCS "src/*.c")

add_executable(train ${SRCS})