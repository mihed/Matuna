FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(OCLConvNetForwardConvolutionTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(OCLConvNetForwardConvolutionTest OCLConvNet ConvNet Math)
SET_TARGET_PROPERTIES(OCLConvNetForwardConvolutionTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(OCLConvNetForwardConvolutionTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "OCLConvNetForwardConvolutionTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

