

CMAKE_MINIMUM_REQUIRED (VERSION 3.2.0)
PROJECT (Matuna.Samples)

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

#Copy the assets to the binary path

FILE(COPY assets DESTINATION ${MAINFOLDER}/bin/)

#Add targets

ADD_SUBDIRECTORY(Matuna.MNISTSample1)