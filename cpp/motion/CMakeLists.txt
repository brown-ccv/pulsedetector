# this is motion



############################################################################
#Executables
############################################################################

add_executable(videostab1 videostab1.cpp)

target_link_libraries(
  videostab1 
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)

############################################################################

add_executable(videostab videostab.cpp)
target_link_libraries(
  videostab 
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)