

CMAKE_MINIMUM_REQUIRED (VERSION 3.2.0)
PROJECT (Matuna.Tests)

SET (MAINFOLDER "${PROJECT_SOURCE_DIR}/..")
SET (EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/bin")
SET (LIBRARY_OUTPUT_PATH "${MAINFOLDER}/lib")

SET (CMAKE_MODULE_PATH "${MAINFOLDER}/share/cmake")

#
# Include the C11 support and compiler warnings
#

INCLUDE(TuneUpCompilerWarnings)
INCLUDE(C11Support)
INCLUDE(OpenCLSupport)

#
# We need to specifiy the standard lib if we compile for Android
#

IF(ANDROID)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++")
ENDIF(ANDROID)

SET (ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/thirdparty/include")
LIST (APPEND ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/include")
IF(ANDROID)
	LIST (APPEND ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/thirdparty/include/Android")
ENDIF(ANDROID)

INCLUDE_DIRECTORIES(${ALL_INCLUDE_DIRECTORIES})

ADD_SUBDIRECTORY(Matuna.OCLDeviceHandlerTest)
ADD_SUBDIRECTORY(Matuna.OCLDeviceTest)
ADD_SUBDIRECTORY(Matuna.ConvNetLayerInterlockTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetInitializationTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetPerceptronTest)
ADD_SUBDIRECTORY(Matuna.MathMatrixTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetStandardOutputLayerTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetBackPropPerceptronTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetGradientPerceptronTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetForwardConvolutionTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetBackConvolutionTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetGradientConvolutionTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetGradientTest)
ADD_SUBDIRECTORY(Matuna.OCLConvNetGDTrainingTest)
ADD_SUBDIRECTORY(Matuna.OCLProgramTest)
ADD_SUBDIRECTORY(Matuna.OCLLayerKernelTest)