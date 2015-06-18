FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.OCLLayerKernelTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.OCLLayerKernelTest Matuna.OCLConvNet Matuna.OCLHelper Matuna.Math Matuna.Helper)
SET_TARGET_PROPERTIES(Matuna.OCLLayerKernelTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.OCLLayerKernelTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.OCLLayerKernelTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

