
cmake_minimum_required(VERSION 2.8)

project(pulse_detector)

#For some reason this doesn’t work for me and I have to manually set them in the GUI
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  MESSAGE("Clang Compiler")
  SET(CMAKE_COMPILER_IS_CLANG 1)
  if("${CMAKE_C_FLAGS}" STREQUAL "")
    SET(CMAKE_C_FLAGS "-Qunused-arguments")
  endif()
  if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    SET(CMAKE_CXX_FLAGS "-ftemplate-depth=1024 -Qunused-arguments -Wno-invalid-offsetof ${SSE_FLAGS}") # Unfortunately older Clang versions do not have this: -Wno-unnamed-type-template-args
  endif()
  MESSAGE("C_XX Flags: ${CMAKE_CXX_FLAGS}")
  SET(CLANG_LIBRARIES "stdc++")
endif()

set(SRCS
  flow_functions.cpp
  )

# OpenCV Config
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

if (OpenCV_FOUND)
	INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
	LINK_DIRECTORIES(${OpenCV_LIB_DIR})

	add_subdirectory(motion)
endif (OpenCV_FOUND)



