FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.OCLConvNetBackConvolutionTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.OCLConvNetBackConvolutionTest Matuna.OCLConvNet Matuna.ConvNet Matuna.Math)
SET_TARGET_PROPERTIES(Matuna.OCLConvNetBackConvolutionTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.OCLConvNetBackConvolutionTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.OCLConvNetBackConvolutionTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

