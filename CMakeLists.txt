cmake_minimum_required(VERSION 3.15)
project(parallel_computing_and_multimedia_processing)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add lab1 executable
add_executable(lab1 srs/lab1/main.cpp)

# Enable OpenMP manually (for Intel Mac with Homebrew)
target_compile_options(lab1 PRIVATE -Xpreprocessor -fopenmp)
target_link_libraries(lab1 PRIVATE /usr/local/opt/libomp/lib/libomp.dylib)
target_include_directories(lab1 PRIVATE /usr/local/opt/libomp/include)

# OpenCV linking
target_include_directories(lab1 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lab1 PRIVATE ${OpenCV_LIBS})
