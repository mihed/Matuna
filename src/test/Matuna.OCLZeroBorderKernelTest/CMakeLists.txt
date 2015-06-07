FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.OCLZeroBorderKernelTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.OCLZeroBorderKernelTest Matuna.OCLConvNet Matuna.ConvNet Matuna.Math)
SET_TARGET_PROPERTIES(Matuna.OCLZeroBorderKernelTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.OCLZeroBorderKernelTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.OCLZeroBorderKernelTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

