/*
 * FileHelper.cpp
 *
 *  Created on: May 11, 2015
 *      Author: Mikael
 */

#include "FileHelper.h"
#include <stdio.h>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>

//Working directory defines
#if defined(_WIN32)
#include <direct.h>
#define GetWorkingDirectory_MACRO _getcwd
#else
#include  <unistd.h>
#define GetWorkingDirectory_MACRO getcwd
#endif

//Executable path defines
#if defined(_WIN32)
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace ATML
{
	namespace Helper
	{
		string FileHelper::GetExecutablePath()
		{
			string result;
			unique_ptr<char[]> rawPath(new char[FILENAME_MAX]);

#if defined(_WIN32)
			DWORD size = sizeof(char) * FILENAME_MAX;
			if (GetModuleFileName(NULL, rawPath.get(), size) != 0)
			{
				result = string(rawPath.get());
				//Convert to platform independent path
				replace(result.begin(), result.end(), '\\', '/');
			}
			else
				throw runtime_error("Could not retrieve the executable path");

#elif defined(__APPLE__)
			auto size = sizeof(char) * FILENAME_MAX;
			if (_NSGetExecutablePath(rawPath.get(), &size) == 0)
				result = string(rawPath);
			else
				throw runtime_error("Could not retrieve the executable path");
#else	//We assume a linux based system here
			char szTmp[32];
			auto size = sizeof(char) * FILENAME_MAX;
			sprintf(szTmp, "/proc/%d/exe", getpid());
			int bytes = min(readlink(szTmp, rawPath.get(), size), size - 1);
			if(bytes >= 0)
			{
				rawPath[bytes] = '\0';
				result = string(rawPath);
			}
			else
				throw runtime_error("Could not retrieve the executable path");

#endif

			return result;
		}

		string FileHelper::GetWorkingDirectory()
		{
			char rawPath[FILENAME_MAX];
			if (!GetWorkingDirectory_MACRO(rawPath, sizeof(rawPath)))
				throw runtime_error("Could not retrieve the working directory.");

			return string(rawPath);
		}

		string FileHelper::GetTextFromPath(const string& path)
		{
			ifstream file(path);
			stringstream stringStream;
			string temp;
			while (getline(file, temp))
				stringStream << temp << endl;

			file.close();

			return stringStream.str();
		}

	} /* namespace Helper */
} /* namespace ATML */
