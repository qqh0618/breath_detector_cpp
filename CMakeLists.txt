cmake_minimum_required(VERSION 3.10)
project(test)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR "D:\\ruanjian\\opencv-4.5.4\\install")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories("${CMAKE_CURRENT_LIST_DIR}")

add_executable(test main.cpp breath_detector.cpp opencv.cpp)
target_link_libraries(test ${OpenCV_LIBS})

#cmake_minimum_required(VERSION 3.10)
#project(code)
#
#set(CMAKE_CXX_STANDARD 14)
#
## Where to find CMake modules and OpenCV
#set(OpenCV_DIR "D:\\ruanjian\\opencv-4.5.4\\install")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
#
#
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#
#add_executable(code main.cpp)
##add_library(code static main.cpp)
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn opencv_plot)
#
#target_link_libraries(code ${OpenCV_LIBS})
