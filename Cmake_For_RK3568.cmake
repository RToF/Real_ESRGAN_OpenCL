cmake_minimum_required(VERSION 3.16)
project(convert)

link_directories(/usr/lib/aarch64-linux-gnu)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_definitions(-DCL_TARGET_OPENCL_VERSION=200)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/usr/include)

add_subdirectory(core)

add_executable(demo bin2C++.cpp)
target_include_directories(demo PRIVATE ${OpenCL_INCLUDE_DIRS})
target_include_directories(demo PRIVATE ./core/include)
target_link_libraries(demo  ${OpenCL_LIBRARIES})
target_link_libraries(demo  ${OpenCV_LIBS})
target_link_libraries(demo  core)
target_link_libraries(demo  mali)
