/*
* IMatunaParsable.h
*
*  Created on: Jun 9, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLHELPER_IMATUNAPARSABLE_H_
#define MATUNA_OCLHELPER_IMATUNAPARSABLE_H_


#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace std;

namespace Matuna
{

	namespace Helper
	{

		class IMatunaParsable
		{
		public:
			//Path of file, with a vector containing the macros and it's substituted value
			virtual unordered_map<string,unordered_map<string, string>> GetDefineSubstitutes() const = 0;
			virtual unordered_map<string, unordered_set<string>> GetDefines() const = 0;
		};

	}
}

#endif /* SOURCE_DIRECTORY__MATUNA_OCLHELPER_IMATUNAPARSABLE_H_ */
