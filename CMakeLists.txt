#------------------------------------------------------------------------------
# Test face alignment algorithms
#------------------------------------------------------------------------------
project(bobetocalo_eccv18)
cmake_minimum_required(VERSION 3.2)

include(${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt)
#include(${CMAKE_CURRENT_LIST_DIR}/../bobetocalo_ert/CMakeLists.txt)

#-- Find include files
set(face_alignment_eccv18_include
  ${faces_framework_include}
  #${face_alignment_bobetocalo_ert_include}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-genfiles
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_alignment_eccv18_src
  ${faces_framework_src}
  #${face_alignment_bobetocalo_ert_src}
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceAlignmentDcfe.cpp
)

set(face_alignment_eccv18_test
  ${CMAKE_CURRENT_LIST_DIR}/test/face_alignment_bobetocalo_eccv18_test.cpp
)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/)

set(face_alignment_eccv18_libs
  ${faces_framework_libs}
  #${face_alignment_bobetocalo_ert_libs}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
  dl
  pthread
)

#-- Setup CMake to run tests
enable_testing()

include_directories(${face_alignment_eccv18_include})

foreach(test ${face_alignment_eccv18_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_alignment_eccv18_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_alignment_eccv18_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name} --database 300w_public)
endforeach()
