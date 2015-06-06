/*
 * AccumulateVectorKernel.h
 *
 *  Created on: Jun 4, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_ACCUMULATEVECTORKERNEL_H_
#define MATUNA_CNNOPENCL_ACCUMULATEVECTORKERNEL_H_

#include "OpenCLHelper/OpenCLKernel.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class AccumulateVectorKernel: public OpenCLKernel
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
public:
	AccumulateVectorKernel(string programName);
	~AccumulateVectorKernel();

	void SetInput(OpenCLMemory* input);
	void SetAccumulator(OpenCLMemory* accumulator);
	void SetGlobalWorkSize(int globalUnits);

	virtual string ProgramName() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOPENCL_ACCUMULATEVECTORKERNEL_H_ */
