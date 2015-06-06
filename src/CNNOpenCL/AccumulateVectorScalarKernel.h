/*
 * AccumulateVectorScalarKernel.h
 *
 *  Created on: Jun 4, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_ACCUMULATEVECTORSCALARKERNEL_H_
#define MATUNA_CNNOPENCL_ACCUMULATEVECTORSCALARKERNEL_H_

#include "OpenCLHelper/OpenCLKernel.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class AccumulateVectorScalarKernel: public OpenCLKernel
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;

public:
	AccumulateVectorScalarKernel(string programName);
	~AccumulateVectorScalarKernel();

	void SetGlobalWorkSize(int globalUnits);
	void SetInput(OpenCLMemory* input);
	void SetAccumulator(OpenCLMemory* accumulator);
	void SetScalar(T scalar);

	virtual string ProgramName() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOPENCL_ACCUMULATEVECTORSCALARKERNEL_H_ */
