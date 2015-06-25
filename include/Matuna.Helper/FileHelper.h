/*
 * FileHelper.h
 *
 *  Created on: May 11, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_HELPER_FILEHELPER_H_
#define MATUNA_MATUNA_HELPER_FILEHELPER_H_

#include <string>

using namespace std;

namespace Matuna
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
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_HELPER_FILEHELPER_H_ */
