FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(OCLConvNetGradientPerceptronTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(OCLConvNetGradientPerceptronTest ConvNetOCL ConvNet Math)
SET_TARGET_PROPERTIES(OCLConvNetGradientPerceptronTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(OCLConvNetGradientPerceptronTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "OCLConvNetGradientPerceptronTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

