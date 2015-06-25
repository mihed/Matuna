/*
 * Path.h
 *
 *  Created on: May 1, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_HELPER_PATH_H_
#define MATUNA_MATUNA_HELPER_PATH_H_

#include <string>
#include <vector>

using namespace std;

namespace Matuna {
namespace Helper {

class Path {
public:
	static string Combine(const string& left, const string& right);
	static string Combine(const string& left, const string& middle, const string& right);
	static string Combine(const vector<string>& strings);
	static string GetDirectoryName(const string& path);
	static string GetDirectoryPath(const string& path);

};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_HELPER_PATH_H_ */
