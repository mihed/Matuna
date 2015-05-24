FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(BackConvolutionKernelTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(BackConvolutionKernelTest CNNOpenCL CNN Math)
SET_TARGET_PROPERTIES(BackConvolutionKernelTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(BackConvolutionKernelTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "BackConvolutionKernelTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

