FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.OCLConvNetInitializationTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.OCLConvNetInitializationTest Matuna.OCLConvNet Matuna.ConvNet Matuna.OCLHelper)
SET_TARGET_PROPERTIES(Matuna.OCLConvNetInitializationTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.OCLConvNetInitializationTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.OCLConvNetInitializationTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

