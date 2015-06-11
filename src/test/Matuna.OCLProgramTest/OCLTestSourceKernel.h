/*
* OCLTestSourceKernel.h
*
*  Created on: Jun 10, 2015
*      Author: Mikael
*/

#ifndef MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLTESTSOURCEKERNEL_H_
#define MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLTESTSOURCEKERNEL_H_

#include "Matuna.OCLHelper/OCLSourceKernel.h"
#include <string>

using namespace Matuna::Helper;

class OCLTestSourceKernel : public OCLSourceKernel
{
private:
	string name;
	vector<size_t> localWorkSize;
	vector<size_t> globalWorkSize;

public:
	OCLTestSourceKernel();
	~OCLTestSourceKernel();

	virtual string Name() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
	virtual vector<string> GetIncludePaths() const override;
	virtual vector<string> GetSourcePaths() const override;
};

#endif /* MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLTESTSOURCEKERNEL_H_ */
