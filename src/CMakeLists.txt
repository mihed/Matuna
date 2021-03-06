# <Copyright Holder>. Copyright (C) <Copyright Year(s)>. <License>.
#
# Project Properties
#
SET (CMAKE_VERBOSE_MAKEFILE 0) # Use 1 for debugging, 0 for release
CMAKE_MINIMUM_REQUIRED (VERSION 3.2.0)
PROJECT (Matuna)

SET (APPLICATION_NAME "Matuna")
SET (APPLICATION_CODENAME "Platon")
SET (APPLICATION_COPYRIGHT_YEARS "2015")
SET (APPLICATION_VERSION_MAJOR 0)
SET (APPLICATION_VERSION_MINOR 1)
SET (APPLICATION_VERSION_PATCH 0)
SET (APPLICATION_VERSION_TYPE SNAPSHOT)
SET (APPLICATION_VERSION_STRING "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}.${APPLICATION_VERSION_PATCH}-${APPLICATION_VERSION_TYPE}")
SET (APPLICATION_VENDOR_ID "com.yourcompany")
SET (APPLICATION_VENDOR_NAME "Your Company")
SET (APPLICATION_VENDOR_URL "yourcompany.com")
SET (APPLICATION_ID "${APPLICATION_VENDOR_ID}.${PROJECT_NAME}")

#
# Project Output Paths
#

SET (MAINFOLDER "${PROJECT_SOURCE_DIR}/..")
SET (CMAKE_MODULE_PATH "${MAINFOLDER}/share/cmake")

#
# Project Search Paths
#

SET (ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/thirdparty/include")
IF(ANDROID)
	LIST (APPEND ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/thirdparty/include/Android")
ENDIF(ANDROID)
LIST (APPEND ALL_INCLUDE_DIRECTORIES "${MAINFOLDER}/src")

INCLUDE_DIRECTORIES(${ALL_INCLUDE_DIRECTORIES})

#
# Add Project Prerequisites 
#

INCLUDE(TuneUpCompilerWarnings)
INCLUDE(C11Support)
INCLUDE(OpenCLSupport)
INCLUDE(AndroidCompilerFlags)
INCLUDE(SetProjectPaths)

#
# Configure Files
#

FILE (GLOB_RECURSE CONFIGINPUTS1 ${MAINFOLDER}/include/*.in.h.cmake)
FILE (GLOB_RECURSE CONFIGINPUTS2 ${MAINFOLDER}/include/*.h.in.cmake)
FILE (GLOB_RECURSE CONFIGINPUTS3 ${MAINFOLDER}/src/*.in.h.cmake)
FILE (GLOB_RECURSE CONFIGINPUTS4 ${MAINFOLDER}/src/*.h.in.cmake)
LIST (APPEND CONFIGINPUTS "${CONFIGINPUTS1}")
LIST (APPEND CONFIGINPUTS "${CONFIGINPUTS2}")
LIST (APPEND CONFIGINPUTS "${CONFIGINPUTS3}")
LIST (APPEND CONFIGINPUTS "${CONFIGINPUTS4}")
FOREACH (CONFIGINPUT ${CONFIGINPUTS})
    STRING (REPLACE ".in.h.cmake" ".h" CONFIGOUTPUT1 "${CONFIGINPUT}")
    STRING (REPLACE ".h.in.cmake" ".h" CONFIGOUTPUT2 "${CONFIGOUTPUT1}")
    CONFIGURE_FILE ("${CONFIGINPUT}" "${CONFIGOUTPUT2}")
ENDFOREACH (CONFIGINPUT ${CONFIGINPUTS})


#
# Moving all source headers to include
#

SET(PROJECT_INCLUDE_DIRECTORY "${MAINFOLDER}/include")

FUNCTION(TraverseAndCopy directory copyDirectory)
	FILE(MAKE_DIRECTORY ${copyDirectory})
	FILE(GLOB MatunaHeaders ${directory}/*.h  ${directory}/*.hpp)
	FILE(COPY ${MatunaHeaders} DESTINATION ${copyDirectory})
	FILE(GLOB children ${directory}/*)	
	FOREACH(child ${children})
		IF(IS_DIRECTORY ${child})
			GET_FILENAME_COMPONENT(childFolderName ${child} NAME)
			SET(newCopyDirectory "${copyDirectory}/${childFolderName}")
			SET(childDirectory "${directory}/${childFolderName}")
			TraverseAndCopy(${childDirectory} ${newCopyDirectory})
		ENDIF()
	ENDFOREACH()
ENDFUNCTION()

FILE(REMOVE_RECURSE ${PROJECT_INCLUDE_DIRECTORY})
TraverseAndCopy(${PROJECT_SOURCE_DIR} ${PROJECT_INCLUDE_DIRECTORY})

#
# Add Build Targets.
#

ADD_SUBDIRECTORY(Matuna.OCLHelper)
ADD_SUBDIRECTORY(Matuna.Helper)
ADD_SUBDIRECTORY(Matuna.Math)
ADD_SUBDIRECTORY(Matuna.ConvNet)
ADD_SUBDIRECTORY(Matuna.OCLConvNet)


#
# Add Documentation Targets
#

INCLUDE (DocumentationTargets)
