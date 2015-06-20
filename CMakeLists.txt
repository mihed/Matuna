# <Copyright Holder>. Copyright (C) <Copyright Year(s)>. <License>.

#
# Debugging Options
#

SET (CMAKE_VERBOSE_MAKEFILE 0) # Use 1 for debugging, 0 for release
#
# Project Properties
#

CMAKE_MINIMUM_REQUIRED (VERSION 3.2.0)
PROJECT(Matuna.ALL)

SET(MATUNA_ALL_IS_USED 1)

#
# Add Build Targets.
#

ADD_SUBDIRECTORY(thirdparty/src)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(samples)
