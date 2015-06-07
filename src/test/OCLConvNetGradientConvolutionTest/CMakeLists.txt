FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.OCLConvNetGradientConvolutionTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.OCLConvNetGradientConvolutionTest Matuna.OCLConvNet Matuna.ConvNet Matuna.Math)
SET_TARGET_PROPERTIES(Matuna.OCLConvNetGradientConvolutionTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.OCLConvNetGradientConvolutionTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.OCLConvNetGradientConvolutionTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

