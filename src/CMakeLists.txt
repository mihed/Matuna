
FILE (GLOB ATML_HEADERS *.h)

# Add Build Targets

#Add installation files

#Copy headers
FILE(COPY ${ATML_HEADERS} DESTINATION ${CMAKE_SOURCE_DIR}/include)