cmake_minimum_required(VERSION 3.0...3.26)

project(vss_test
    LANGUAGES CXX)

include_directories(src)

set(CMAKE_CXX_STANDARD 17)

# find_package(OpenMP REQUIRED)
# find_package(faiss REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "-O0 -g -std=c++17 -DHAVE_CXX0X -fpic -ftree-vectorize")
else()
    set(CMAKE_CXX_FLAGS "-Ofast -std=c++17 -DHAVE_CXX0X -march=native -fpic -w -ftree-vectorize")
endif()

add_executable(vss_test vss_test.cpp)
# target_link_libraries(vss_test faiss)

# add_executable(faiss_test faiss_test.cpp)
# target_link_libraries(faiss_test faiss)