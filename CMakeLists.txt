
cmake_minimum_required(VERSION 2.8)

project(pibilight)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 3 REQUIRED)

include_directories( 
    include
    ${OpenCV_INCLUDE_DIRS} 
)

add_executable( ${PROJECT_NAME} 
    src/pibilight.cpp 
    src/ext/loguru.cpp
)

target_link_libraries( ${PROJECT_NAME} 
    ${OpenCV_LIBS}
    pthread
    dl
)
