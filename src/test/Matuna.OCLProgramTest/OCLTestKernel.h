/*
* OCLTestKernel.h
*
*  Created on: Jun 10, 2015
*      Author: Mikael
*/

#ifndef MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLTESTKERNEL_H_
#define MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLTESTKERNEL_H_

#include "Matuna.OCLHelper/OCLKernel.h"

#include <string>

using namespace std;
using namespace Matuna::Helper;

class OCLTestKernel : public OCLKernel
{
private:
	string name;
	vector<size_t> localWorkSize;
	vector<size_t> globalWorkSize;

public:
	OCLTestKernel(string name);
	~OCLTestKernel();

	virtual string Name() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

#endif /* MATUNA_TEST_MATUNA_OCLPROGRAMTEST_OCLTESTKERNEL_H_ */
