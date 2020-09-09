cmake_minimum_required(VERSION 3.16)
project(kbot_mxnet C CXX)

set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(KBOT_NATIVE_MAJOR=1 KBOT_NATIVE_MINOR=0 KBOT_NATIVE_PATCH=0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Mxnet REQUIRED)
find_package(JNI REQUIRED)

# JNI
include_directories(${JNI_INCLUDE_DIRS})
link_libraries(${JNI_LIBRARIES})

# MXNet
add_compile_definitions(MXNET_USE_CUDA=0 MXNET_USE_BLAS_OPEN=1 MSHADOW_STAND_ALONE=1)
include_directories(${MXNET_INCLUDE_DIRS})
link_libraries(${MXNET_LIBRARIES})

add_library(kbot_mxnet SHARED src/kbot_mxnet_MXNetwork.cpp src/kbot_mxnet_MXNetwork.h src/MXNetwork.cpp src/MXNetwork.h)
target_link_libraries(kbot_mxnet pthread)

add_executable(mxnet_test_train1D src/mxnet_test_train1D.cpp)
target_link_libraries(mxnet_test_train1D kbot_mxnet)

add_executable(mxnet_test_train2D src/mxnet_test_train2D.cpp)
target_link_libraries(mxnet_test_train2D kbot_mxnet)

add_executable(mxnet_test_leaks src/mxnet_test_leaks.cpp)
target_link_libraries(mxnet_test_leaks kbot_mxnet)

# installs to /usr/lib/jni where java picks it up by default
install(TARGETS kbot_mxnet DESTINATION /usr/lib/jni)
add_custom_target(uninstall COMMAND rm /usr/lib/jni/libkbot_mxnet.so)