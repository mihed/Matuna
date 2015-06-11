/*
* OCLMatunaTestKernel.h
*
*  Created on: Jun 10, 2015
*      Author: Mikael
*/

#ifndef MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLMATUNATESTKERNEL_H_
#define MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLMATUNATESTKERNEL_H_

#include "Matuna.OCLHelper/IMatunaParsable.h"
#include "Matuna.OCLHelper/OCLSourceKernel.h"

using namespace Matuna::Helper;

class OCLMatunaTestKernel: public IMatunaParsable, public OCLSourceKernel
{
private:
	string name;
	vector<size_t> localWorkSize;
	vector<size_t> globalWorkSize;
	vector<string> sourcePaths;
	vector<string> includePaths;
	unordered_map<string,unordered_map<string, string>> subs;
	unordered_map<string, unordered_set<string>> defines;

public:
	OCLMatunaTestKernel();
	~OCLMatunaTestKernel();

	virtual string Name() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
	virtual vector<string> GetIncludePaths() const override;
	virtual vector<string> GetSourcePaths() const override;
	virtual unordered_map<string,unordered_map<string, string>> GetDefineSubstitutes() const override;
	virtual unordered_map<string, unordered_set<string>> GetDefines() const override;
};

#endif /* MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLMATUNATESTKERNEL_H_ */
