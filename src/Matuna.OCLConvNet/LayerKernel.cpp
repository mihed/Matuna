/*
* LayerKernel.cpp
*
*  Created on: Jun 11, 2015
*      Author: Mikael
*/

#include "LayerKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		LayerKernel<T>::LayerKernel()
		{

		}

		template<class T>
		LayerKernel<T>::~LayerKernel()
		{

		}

		template<class T>
		void LayerKernel<T>::SetKernelName(string name)
		{
			this->name = name;
		}

		template<class T>
		void LayerKernel<T>::AddDefineSubsitute(string path, string defineName, string defineValue)
		{
			if (substitutes.find(path) == substitutes.end())
				substitutes.insert(make_pair(path, unordered_map<string, string>()));

			auto& map = substitutes[path];
			if (map.find(defineName) != map.end())
				throw invalid_argument("A define with that name has already been added");

			map.insert(make_pair(defineName, defineValue));
		}

		template<class T>
		void LayerKernel<T>::AddDefine(string path, string defineName)
		{
			if (defines.find(path) == defines.end())
				defines.insert(make_pair(path, unordered_set<string>()));

			auto& set = defines[path];
			if (set.find(defineName) != set.end())
				throw invalid_argument("The define has already been added");

			set.insert(defineName);
		}

		template<class T>
		void LayerKernel<T>::ClearGlobalSizes()
		{
			globalWorkSize.clear();
		}

		template<class T>
		void LayerKernel<T>::ClearLocalSizes()
		{
			localWorkSize.clear();
		}

		template<class T>
		void LayerKernel<T>::SetMemoryArg(OCLMemory* memory, int index)
		{
			if (!this->KernelSet())
				throw runtime_error("The kernel has not been attached");

			auto rawMemory = memory->GetCLMemory();
			CheckOCLError(
				clSetKernelArg(this->GetKernel(), index, sizeof(cl_mem), &rawMemory),
				"Could not set the kernel argument");
		}

		template<class T>
		void LayerKernel<T>::SetRealArg(T value, int index)
		{
			if (!this->KernelSet())
				throw runtime_error("The kernel has not been attached");

			CheckOCLError(
				clSetKernelArg(this->GetKernel(), index, sizeof(T), &value),
				"Could not set the kernel argument");
		}

		template<class T>
		void LayerKernel<T>::AddIncludePath(string name)
		{
			bool include = true;
			for (auto& path : includePaths)
				if (path.compare(name) == 0)
					include = false;
			if (include)
				includePaths.push_back(name);
		}

		template<class T>
		void LayerKernel<T>::AddSourcePath(string name)
		{
			bool include = true;
			for (auto& path : sourcePaths)
				if (path.compare(name) == 0)
					include = false;
			if (include)
				sourcePaths.push_back(name);
		}

		template<class T>
		void LayerKernel<T>::SetLocalMemoryArg(size_t size, int index)
		{
			if (!this->KernelSet())
				throw runtime_error("The kernel has not been attached");

			CheckOCLError(
				clSetKernelArg(this->GetKernel(), index, size, nullptr),
				"Could not set the kernel argument");
		}


		template<class T>
		string LayerKernel<T>::Name() const
		{
			return name;
		}

		template<class T>
		const vector<size_t>& LayerKernel<T>::GlobalWorkSize() const
		{
			return globalWorkSize;
		}

		template<class T>
		const vector<size_t>& LayerKernel<T>::LocalWorkSize() const
		{
			return localWorkSize;
		}

		template<class T>
		vector<string> LayerKernel<T>::GetIncludePaths() const
		{
			return includePaths;
		}

		template<class T>
		vector<string> LayerKernel<T>::GetSourcePaths() const
		{
			return sourcePaths;
		}

		template<class T>
		unordered_map<string,unordered_map<string, string>> LayerKernel<T>::GetDefineSubstitutes() const
		{
			return substitutes;
		}

		template<class T>
		unordered_map<string, unordered_set<string>> LayerKernel<T>::GetDefines() const
		{
			return defines;
		}

		template<class T>
		void LayerKernel<T>::AddGlobalSize(size_t size)
		{
			if (globalWorkSize.size() > 3)
				throw invalid_argument("The global work size cannot be greater than 3");

			globalWorkSize.push_back(size);
		}

		template<class T>
		void LayerKernel<T>::AddLocalSize(size_t size)
		{
			if (localWorkSize.size() > 3)
				throw invalid_argument("The local work size cannot be greater than 3");

		}

		template class LayerKernel<cl_float>;
		template class LayerKernel<cl_double>;

	} /* namespace MachineLearning */
} /* namespace Matuna */
