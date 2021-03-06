cmake_minimum_required(VERSION 2.8.7)

# faiss project
project(faiss C CXX)

option(BUILD_TUTORIAL "Build tutorials" ON)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_WITH_GPU "Build faiss with gpu (cuda) support" ON)
option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# OpenMP
find_package(OpenMP REQUIRED)

# BLAS (MKL os OpenBLAS)
if(WITH_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    set(BLAS_LIB ${MKL_LIBRARIES})
	LINK_DIRECTORIES(${MKL_LIB_DIR_P} )
	message("mkl!!! ${MKL_LIB_DIR_P}  ${MKL_COMPILER_LIBRARIES} ${MKL_INCLUDE_DIRS} ${MKL_LIBRARIES}")
else()
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIB ${OpenBLAS_LIB})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -m64 -Wall -mavx2 -mpopcnt -fopenmp -Wno-sign-compare -Wno-unused-variable -Wno-unused-function")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_definitions(-DFINTEGER=int)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# specify header and cpp files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB faiss_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB faiss_cpu_impl_cpp ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.cpp)
file(GLOB faiss_cpu_utils_cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)

set(faiss_lib faiss)
add_library(${faiss_lib} SHARED ${faiss_cpu_cpp} ${faiss_cpu_impl_cpp} ${faiss_cpu_utils_cpp})
target_link_libraries(${faiss_lib} ${OpenMP_CXX_FLAGS} ${BLAS_LIB})

# build gpu lib
if(BUILD_WITH_GPU)
    include(cmake/Cuda.cmake)
    add_subdirectory(gpu)
endif(BUILD_WITH_GPU)

# build tutorial examples
#if(BUILD_TUTORIAL)
    add_subdirectory(tutorial)
#endif(BUILD_TUTORIAL)

# build tests
if(BUILD_TEST)
    add_subdirectory(tests)
endif(BUILD_TEST)
