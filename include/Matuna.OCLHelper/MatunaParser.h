/*
 * MatunaParser.h
 *
 *  Created on: Jun 9, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLHELPER_MATUNAPARSER_H_
#define MATUNA_OCLHELPER_MATUNAPARSER_H_

#include "IMatunaParsable.h"
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

namespace Matuna
{
namespace Helper
{

class MatunaParser
{
private:
	static const string beginTag;
	static const string endTag;
	static const string commentTag;
	static const string defineTag;
public:
	MatunaParser();
	~MatunaParser();

	unordered_map<string, string> Parse(IMatunaParsable* element, vector<string> sourcePaths);
};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_MATUNAPARSER_H_ */
