/*
 * FileHelper.h
 *
 *  Created on: May 11, 2015
 *      Author: Mikael
 */

#ifndef ATML_HELPER_FILEHELPER_H_
#define ATML_HELPER_FILEHELPER_H_

#include <string>

using namespace std;

namespace ATML
{
namespace Helper
{

class FileHelper
{
public:
	static string GetExecutablePath();
	static string GetWorkingDirectory();
	static string GetTextFromPath(const string& path);
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_HELPER_FILEHELPER_H_ */
