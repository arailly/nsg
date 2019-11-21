cmake_minimum_required(VERSION 3.14)
project(nsg)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)

set(ARAILIB_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../arailib/include)
set(JSON_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../json/single_include)
set(RANGE_SEARCH_INCLUDE_PATH ${PROJECT_SOURCE_DIR})
include_directories(${ARAILIB_INCLUDE_PATH} ${NNDESCENT_INCLUDE_PATH} ${JSON_INCLUDE_PATH} ${RANGE_SEARCH_INCLUDE_PATH})

add_subdirectory(test)

# omp option
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
set(OpenMP_C_LIB_NAMES omp)
set(OpenMP_CXX_LIB_NAMES omp)
set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Xpreprocessor -fopenmp -O3")

target_include_directories(${PROJECT_NAME} PRIVATE /usr/local/opt/libomp/include)
target_link_libraries(${PROJECT_NAME} PRIVATE /usr/local/opt/libomp/lib/libomp.dylib)

find_package(OpenMP)
if(OpenMP_FOUND)
    message("OpenMP Found!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()