/*
 * OCLSourceKernel.h
 *
 *  Created on: Jun 9, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLHELPER_OCLSOURCEKERNEL_H_
#define MATUNA_OCLHELPER_OCLSOURCEKERNEL_H_

#include "OCLKernel.h"

namespace Matuna
{
namespace Helper
{

class OCLSourceKernel: public OCLKernel
{
public:
	OCLSourceKernel();
	virtual ~OCLSourceKernel();

	virtual vector<string> GetIncludePaths() const = 0;
	virtual vector<string> GetSourcePaths() const = 0;
};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_OCLSOURCEKERNEL_H_ */
