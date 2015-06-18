FILE (GLOB_RECURSE sources_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)

ADD_EXECUTABLE(Matuna.ConvNetLayerInterlockTest ${sources_SRCS})
TARGET_LINK_LIBRARIES(Matuna.ConvNetLayerInterlockTest Matuna.ConvNet)
SET_TARGET_PROPERTIES(Matuna.ConvNetLayerInterlockTest PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(Matuna.ConvNetLayerInterlockTest PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME "Matuna.ConvNetLayerInterlockTest" CLEAN_DIRECT_OUTPUT 1)

#Install directives

