cmake_minimum_required(VERSION 2.4.6)

set(CMAKE_BUILD_TYPE Debug)

#include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
FIND_PACKAGE(CUDA)
#FIND_PACKAGE(OpenCV REQUIRED)
#FIND_PACKAGE(Qt5Widgets REQUIRED)
#FIND_PACKAGE( OpenMP )

include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV manually installed.
include_directories("/cluster/home/czuidema/include")
link_directories("/cluster/home/czuidema/lib")
include_directories("/cluster/home/czuidema/bin")

link_directories("$ENV{CUDA_ROOT}/lib64")
link_directories("$ENV{CUDA_ROOT}/lib64/stubs")


set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS filesystem) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    add_executable(flow ../src/flow_main.cpp) 
    target_link_libraries(flow ${Boost_LIBRARIES})
endif()


# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

SET(PROJECT_NAME superresolution)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("$ENV{CUDADIR}/include")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -O2")
SET(CMAKE_CXX_FLAGS "$ENV{CMAKE_CXX_FLAGS} -Wall -O3")
SET(CMAKE_C_FLAGS "$ENV{CMAKE_C_FLAGS} -Wall -O3")

set(CPU_LIBRARY_TYPE "")
set(GPU_LIBRARY_TYPE "")


# UNCOMMENTED
set(CUDA_ARCH_FLAGS 
#-arch $ENV{CUDA_ARCH}
-arch sm_61
# -D$ENV{CUDA_CAPABILITY} -arch $ENV{CUDA_ARCH}
)

set(CUDAOPTIONS 
#-ptxas-options=-v -DUNIX -g -G
--ptxas-options=-v -DUNIX
${CUDA_ARCH_FLAGS}
)
#

add_library(linearoperations ${CPU_LIBRARY_TYPE}
src/linearoperations/linearoperations.cpp
)
cuda_add_library(linearoperationsGPU ${GPU_LIBRARY_TYPE}
src/linearoperations/linearoperations.cu
OPTIONS ${CUDAOPTIONS}
)

add_library(imagepyramid ${CPU_LIBRARY_TYPE}
src/imagepyramid/imagepyramidcpu.cpp
)
cuda_add_library(imagepyramidGPU ${GPU_LIBRARY_TYPE}
src/imagepyramid/imagepyramidgpu.cpp
OPTIONS ${CUDAOPTIONS}
)

add_library(flowlib ${CPU_LIBRARY_TYPE}
src/flowlib/flowio.cpp
src/flowlib/flowlib.cpp
src/flowlib/flowlib_cpu_sor.cpp
)
cuda_add_library(flowlibGPU ${GPU_LIBRARY_TYPE}
src/flowlib/flowlib_gpu_sor.cu
src/flowlib/flowlib.cu
OPTIONS ${CUDAOPTIONS}
)

add_library(filesystem ${CPU_LIBRARY_TYPE}
src/filesystem/filesystem.cpp
)

add_library(auxiliaryCPU ${CPU_LIBRARY_TYPE}
src/auxiliary/debug.cpp
)
cuda_add_library(auxiliaryGPU ${GPU_LIBRARY_TYPE}
src/auxiliary/cuda_basic.cu
OPTIONS ${CUDAOPTIONS}
)

add_library(superresolutionlib ${CPU_LIBRARY_TYPE}
src/superresolution/superresolution.cpp
)

cuda_add_library(superresolutionlibGPU ${GPU_LIBRARY_TYPE}
src/superresolution/superresolution.cu
OPTIONS ${CUDAOPTIONS}
)

add_executable(superresolution
src/superresolution_main.cpp
)

#link_directories(${CUDA_ROOT}/lib64/stubs)
#link_directories(${CUDA_ROOT}/lib64/)


target_link_libraries(superresolution
superresolutionlib superresolutionlibGPU
flowlib flowlibGPU
imagepyramid imagepyramidGPU
linearoperations linearoperationsGPU
filesystem
auxiliaryCPU auxiliaryGPU
opencv_core opencv_highgui opencv_imgproc
#${OpenCV_LIBS}
${CUDA_LIBRARIES}
cuda cudart cusparse
)


# UNCOMMENT IF BOOST FOUND
#add_executable(flow
#src/flow_main.cpp
#)

target_link_libraries(flow
flowlib flowlibGPU
imagepyramid imagepyramidGPU
linearoperations linearoperationsGPU
auxiliaryCPU auxiliaryGPU
opencv_core opencv_highgui opencv_imgproc 
#${OpenCV_LIBS}
${CUDA_LIBRARIES}
cuda cudart cusparse
#Qt5::Widgets
)

add_executable(memcmp src/memcmp.cpp)
target_link_libraries(memcmp opencv_core opencv_highgui)

add_executable(memcmptest src/memcmptest.cpp)
target_link_libraries(memcmptest opencv_core)


