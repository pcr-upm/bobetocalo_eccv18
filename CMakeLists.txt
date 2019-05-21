#------------------------------------------------------------------------------
# Test face alignment algorithms
#------------------------------------------------------------------------------
project(bobetocalo_eccv18)
cmake_minimum_required(VERSION 3.2)

include(${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/../ert_simple/CMakeLists.txt)

#-- Setup required libraries
find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS FOUND_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11")
else()
  message(STATUS NOT_FOUND_OPENMP)
endif()

#-- Find include files
set(face_alignment_bobetocalo_eccv18_include
  ${faces_framework_include}
  ${face_alignment_ert_simple_include}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-genfiles
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_alignment_bobetocalo_eccv18_src
  ${faces_framework_src}
  ${face_alignment_ert_simple_src}
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceAlignment3dde.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/HonariChannelFeatures.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/GaussianChannelFeatures.cpp
)

set(face_alignment_bobetocalo_eccv18_test
  ${CMAKE_CURRENT_LIST_DIR}/test/face_alignment_bobetocalo_eccv18_test.cpp
)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/)

set(face_alignment_bobetocalo_eccv18_libs
  ${faces_framework_libs}
  ${face_alignment_ert_simple_libs}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
  dl
  pthread
)

#-- Setup CMake to run tests
enable_testing()

include_directories(${face_alignment_bobetocalo_eccv18_include})

foreach(test ${face_alignment_bobetocalo_eccv18_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_alignment_bobetocalo_eccv18_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_alignment_bobetocalo_eccv18_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name} --database 300w_public --measure pupils)
endforeach()
