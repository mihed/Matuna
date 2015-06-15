/*
* LayerKernel.h
*
*  Created on: Jun 11, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLCONVNET_LAYERKERNEL_H_
#define MATUNA_MATUNA_OCLCONVNET_LAYERKERNEL_H_

#include "Matuna.OCLHelper/OCLMemory.h"
#include "Matuna.OCLHelper/OCLSourceKernel.h"
#include "Matuna.OCLHelper/IMatunaParsable.h"
#include <string>
#include <vector>

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
	namespace MachineLearning
	{
		template<class T>
		class LayerKernel: public IMatunaParsable, public OCLSourceKernel
		{
		private:
			string name;
			vector<size_t> localWorkSize;
			vector<size_t> globalWorkSize;
			vector<string> sourcePaths;
			vector<string> includePaths;
			unordered_map<string,unordered_map<string, string>> substitutes;
			unordered_map<string, unordered_set<string>> defines;
		public:
			LayerKernel();
			~LayerKernel();

			void SetKernelName(string name);
			void AddDefineSubsitute(string path, string defineName, int defineValue);
			void AddDefineSubsitute(string path, string defineName, float defineValue);
			void AddDefineSubsitute(string path, string defineName, double defineValue);
			void AddDefineSubsitute(string path, string defineName, long defineValue);
			void AddDefine(string path, string defineName);

			void SetMemoryArg(OCLMemory* memory, int index);
			void SetRealArg(T value, int index);
			void SetLocalMemoryArg(size_t size, int index);

			void AddIncludePath(string path);
			void AddSourcePath(string path);

			void AddGlobalSize(size_t size);
			void AddLocalSize(size_t size);

			void ClearGlobalSizes();
			void ClearLocalSizes();

			virtual string Name() const override;
			virtual const vector<size_t>& GlobalWorkSize() const override;
			virtual const vector<size_t>& LocalWorkSize() const override;
			virtual vector<string> GetIncludePaths() const override;
			virtual vector<string> GetSourcePaths() const override;
			virtual unordered_map<string,unordered_map<string, string>> GetDefineSubstitutes() const override;
			virtual unordered_map<string, unordered_set<string>> GetDefines() const override;

		private:
			void InsertDefineSubstitute(string path, string defineName, string value);
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_LAYERKERNEL_H_ */
