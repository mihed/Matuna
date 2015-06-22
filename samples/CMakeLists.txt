

SET (CMAKE_VERBOSE_MAKEFILE 0) # Use 1 for debugging, 0 for release
CMAKE_MINIMUM_REQUIRED (VERSION 3.2.0)
PROJECT (Matuna.Samples)

SET (MAINFOLDER "${PROJECT_SOURCE_DIR}/..")
SET (CMAKE_MODULE_PATH "${MAINFOLDER}/share/cmake")

INCLUDE(TuneUpCompilerWarnings)
INCLUDE(C11Support)
INCLUDE(OpenCLSupport)
INCLUDE(AndroidCompilerFlags)
INCLUDE(SetProjectPaths)

SET (ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/thirdparty/include")
LIST (APPEND ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/include")
LIST (APPEND ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/samples")
IF(ANDROID)
	LIST (APPEND ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/thirdparty/include/Android")
ENDIF(ANDROID)

INCLUDE_DIRECTORIES(${ALL_INCLUDE_DIRECTORIES})

#Add targets


MACRO(SetSource folderName)
	FILE (GLOB_RECURSE sources_SRCS ${folderName}/*.cpp ${folderName}/*.cxx ${folderName}/*.cc ${folderName}/*.C ${folderName}/*.c ${folderName}/*.h ${folderName}/*.hpp)
ENDMACRO()

MACRO(SetStandardSample testName)
	SetSource("${CMAKE_CURRENT_LIST_DIR}/${testName}")
	ADD_EXECUTABLE(${testName} ${sources_SRCS})
	TARGET_LINK_LIBRARIES(${testName} Matuna.OCLConvNet Matuna.OCLHelper ${OCL_LIBRARY} Matuna.ConvNet Matuna.Helper Matuna.Math)
	SET_TARGET_PROPERTIES(${testName} PROPERTIES LINKER_LANGUAGE CXX)
	SET_TARGET_PROPERTIES(${testName} PROPERTIES OUTPUT_NAME ${testName} CLEAN_DIRECT_OUTPUT 1)
	
	ADD_CUSTOM_COMMAND(
	TARGET ${testName} 
	POST_BUILD
	COMMAND ${CMAKE_COMMAND}
	ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/assets ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}/assets
	)

	ADD_CUSTOM_COMMAND(
	TARGET ${testName}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND}
	ARGS -E copy_directory ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}/kernels ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}/kernels
	)
ENDMACRO()

SetStandardSample(Matuna.MNISTSample1)
SetStandardSample(Matuna.MNISTSample2)