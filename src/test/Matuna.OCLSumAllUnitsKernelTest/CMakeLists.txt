FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.OCLSumAllUnitsKernelTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.OCLSumAllUnitsKernelTest Matuna.OCLConvNet Matuna.ConvNet Matuna.Math)
SET_TARGET_PROPERTIES(Matuna.OCLSumAllUnitsKernelTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.OCLSumAllUnitsKernelTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.OCLSumAllUnitsKernelTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

