FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(OCLConvNetGradientConvolutionTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(OCLConvNetGradientConvolutionTest ConvNetOCL ConvNet Math)
SET_TARGET_PROPERTIES(OCLConvNetGradientConvolutionTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(OCLConvNetGradientConvolutionTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "OCLConvNetGradientConvolutionTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

