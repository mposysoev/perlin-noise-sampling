cmake_minimum_required(VERSION 3.0.0)
project(perlin-noise-texture VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# Set the path to the OpenCV installation
set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.7.0_6/)

# Find OpenCV package
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(perlin-noise-texture main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${OpenCV_INCLUDE_DIRS})

# Add OpenCV libraries to your project
target_link_libraries(perlin-noise-texture ${OpenCV_LIBS})