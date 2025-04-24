cmake_minimum_required(VERSION 3.10)
project(parallel_computing_and_multimedia_processing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Add future labs here
add_executable(lab1 sis/lab1/main.cpp)
target_link_libraries(lab1 ${OpenCV_LIBS})
