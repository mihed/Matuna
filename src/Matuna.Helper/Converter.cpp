/*
 * Converter.cpp
 *
 *  Created on: Jun 15, 2015
 *      Author: mikael
 */

#include "Converter.h"
#include <sstream>

namespace Matuna
{
namespace Helper
{
string Converter::ConvertToString(int value)
{
	stringstream stream;
	stream << value;
	return stream.str();
}
string Converter::ConvertToString(float value)
{
	stringstream stream;
	stream << value;
	return stream.str();
}
string Converter::ConvertToString(double value)
{
	stringstream stream;
	stream << value;
	return stream.str();
}

string Converter::ConvertToString(long value)
{
	stringstream stream;
	stream << value;
	return stream.str();
}

} /* namespace Helper */
} /* namespace Matuna */
