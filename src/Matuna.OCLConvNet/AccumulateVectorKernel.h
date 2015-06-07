/*
 * AccumulateVectorKernel.h
 *
 *  Created on: Jun 4, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_ACCUMULATEVECTORKERNEL_H_
#define MATUNA_OCLConvNet_ACCUMULATEVECTORKERNEL_H_

#include "Matuna.OCLHelper/OCLKernel.h"
#include "Matuna.OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class AccumulateVectorKernel: public OCLKernel
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
public:
	AccumulateVectorKernel(string programName);
	~AccumulateVectorKernel();

	void SetInput(OCLMemory* input);
	void SetAccumulator(OCLMemory* accumulator);
	void SetGlobalWorkSize(int globalUnits);

	virtual string ProgramName() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_OCLConvNet_ACCUMULATEVECTORKERNEL_H_ */
