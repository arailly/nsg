cmake_minimum_required(VERSION 3.14)

add_executable(test test.cpp)

target_link_libraries(test gtest gtest_main /usr/local/opt/libomp/lib/libomp.dylib)
target_include_directories(test PRIVATE ${PROJECT_SOURCE_DIR}/include)

set(ARAILIB_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../arailib/include)
target_include_directories(test PRIVATE ${ARAILIB_INCLUDE_PATH})

# omp option
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_LIB_NAMES omp)
set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Xpreprocessor -fopenmp")

target_include_directories(test PRIVATE /usr/local/opt/libomp/include)

find_package(OpenMP)
if(OpenMP_FOUND)
    message("OpenMP Found!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()