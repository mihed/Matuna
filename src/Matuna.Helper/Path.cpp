/*
 * Path.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Mikael
 */

#include "Path.h"

namespace Matuna {
namespace Helper {

string Path::Combine(const string& left, const string& right) {

	auto leftSize = left.size();
	auto rightSize = right.size();
	if (leftSize == 0)
		return right;
	else if (rightSize == 0)
		return left;

	if (left[leftSize - 1] == '/') {
		if (right[0] == '/')
			return left + right.substr(1, rightSize - 1);
		else
			return left + right;
	} else {
		if (right[0] == '/')
			return left + right;
		else
			return left + "/" + right;
	}
}

string Path::Combine(const string& left, const string& middle,
		const string& right) {
	return Combine(Combine(left, middle), right);
}

string Path::Combine(const vector<string>& strings) {
	string result;
	auto count = strings.size();

	if (count == 0)
		return result;
	result = strings[0];
	for(size_t i = 1; i < count; i++)
		result += Combine(result,strings[i]);

	return result;
}

string Path::GetDirectoryName(const string& path) {
	vector<size_t> positions;
	size_t position = path.find("/");
	while(position != string::npos)
	{
		positions.push_back(position);
		position = path.find("/", position + 1);
	}

	auto count = positions.size();
	if (count == 0)
		return string();
	else if (count == 1)
		return path.substr(0, positions[0] + 1);
	else
		return path.substr(positions[count - 2], positions[count - 1] - positions[count - 2]);
}

string Path::GetDirectoryPath(const string& path)
{
	size_t position = path.find_last_of("/");
	return path.substr(0, position);
}

} /* namespace Helper */
} /* namespace Matuna */
