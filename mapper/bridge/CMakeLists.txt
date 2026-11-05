cmake_minimum_required(VERSION 3.20)
project(mapper_bridge CXX)

# Builds the bridge against the RTAB-Map that is installed (headers + rtabmap_core/utilite import libs) and the OpenCV 4.7.0 headers and
# import libraries prepared in ../deps by tools/fetch_deps.py and tools/make_importlibs.py. No find_package: the installed package would also ask for PCL/Qt development files.
set(RTABMAP_DIR "C:/Program Files/RTABMap" CACHE PATH "installed RTAB-Map")
set(DEPS "${CMAKE_CURRENT_SOURCE_DIR}/../../rtabmap_lab/deps")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
foreach(cfg RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
endforeach()

file(GLOB CV_LIBS "${DEPS}/implibs/opencv_*470.lib")
add_executable(mapper_bridge main.cpp)
target_include_directories(mapper_bridge SYSTEM PRIVATE
  "${RTABMAP_DIR}/include/rtabmap-0.23" "${DEPS}/opencv/include" "${DEPS}/eigen" "${DEPS}/stubs")
target_link_directories(mapper_bridge PRIVATE "${RTABMAP_DIR}/lib")
target_link_libraries(mapper_bridge PRIVATE rtabmap_core rtabmap_utilite ${CV_LIBS} ws2_32)
target_compile_definitions(mapper_bridge PRIVATE _USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(mapper_bridge PRIVATE /EHsc /bigobj /W3 /utf-8)
