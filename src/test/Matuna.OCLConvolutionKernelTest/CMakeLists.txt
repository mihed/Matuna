FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.OCLConvolutionKernelTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.OCLConvolutionKernelTest Matuna.OCLConvNet Matuna.ConvNet Matuna.Math)
SET_TARGET_PROPERTIES(Matuna.OCLConvolutionKernelTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.OCLConvolutionKernelTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.OCLConvolutionKernelTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

