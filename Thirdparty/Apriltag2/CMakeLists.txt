cmake_minimum_required(VERSION 2.6)
project(Apriltag2)

add_custom_target(Apriltag2 ALL COMMAND make 
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apriltag_src
)
