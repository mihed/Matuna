/*
 * Converter.h
 *
 *  Created on: Jun 15, 2015
 *      Author: mikael
 */

#ifndef MATUNA_MATUNA_HELPER_CONVERTER_H_
#define MATUNA_MATUNA_HELPER_CONVERTER_H_

#include <string>

using namespace std;

namespace Matuna
{
namespace Helper
{

class Converter
{
public:
	static string ConvertToString(int);
	static string ConvertToString(float);
	static string ConvertToString(double);
	static string ConvertToString(long);
};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_HELPER_CONVERTER_H_ */
