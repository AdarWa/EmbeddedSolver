cmake_minimum_required(VERSION 3.16)
project(EmbeddedSolverTraining C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Get all source files, excluding main.c
file(GLOB_RECURSE LIB_SRCS "src/*.c")
list(FILTER LIB_SRCS EXCLUDE REGEX "src/main.c")

# Compile main.c and all source files directly into the train executable
add_executable(train src/main.c ${LIB_SRCS})
target_include_directories(train PRIVATE src)

enable_testing()
include(FetchContent)
include(CTest)

FetchContent_Declare(
        unity
        GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
        GIT_TAG        v2.5.2
)
FetchContent_MakeAvailable(unity)
target_compile_definitions(unity PUBLIC UNITY_INCLUDE_DOUBLE)

file(GLOB TEST_FILES "tests/test_*.c")
message(STATUS "FOUND TEST FILES: ${TEST_FILES}")

foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)

    # Compile the test file and all source files directly into the test executable
    add_executable(${test_name} ${test_file} ${LIB_SRCS})

    # Add the src directory so the test can find the headers
    target_include_directories(${test_name} PRIVATE src)

    # Link only Unity, since the source files are already compiled in
    target_link_libraries(${test_name} PRIVATE unity)

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()