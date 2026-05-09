# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.22)

if(EXISTS "$ENV{IDF_PATH}/tools/cmake/project.cmake" AND NOT DESKTOP_BUILD)

    # Define which directories ESP-IDF should ignore
    # This prevents 'train' from being treated as an ESP-IDF component
    set(EXTRA_COMPONENT_DIRS "")
    set(COMPONENTS main managed_components) # Explicitly list ESP components

    include($ENV{IDF_PATH}/tools/cmake/project.cmake)
    project(EmbeddedSolver)

else()
    # 2. Desktop Build Logic
    project(EmbeddedSolverTraining)
    enable_testing()
    add_subdirectory(train)
endif()
