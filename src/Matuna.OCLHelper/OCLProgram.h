/*
* OCLProgram.h
*
*  Created on: Jun 9, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLHELPER_OCLPROGRAM_H_
#define MATUNA_OCLHELPER_OCLPROGRAM_H_

#include "OCLKernel.h"
#include "OCLInclude.h"
#include "OCLSourceKernel.h"
#include "IMatunaParsable.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

using namespace std;

namespace Matuna {
	namespace Helper {
		class OCLProgram {

			friend class OCLContext;

		private:
			unordered_map<string, unique_ptr<OCLKernel>> kernels;
			const OCLContext* context;
			cl_program program;
			string name;

			static int instanceCounter;

			bool useRelaxedMath;
			bool enableMAD;
			bool enableMatunaScript;

			unordered_set<string> defines;
			unordered_map<string, string> definesWithValues;

			unordered_set<string> includePaths;
			unordered_map<string, string> pathAndCodeFiles;

		public:
			static string DefaultSourceLocation;
			static const string MADCompilerOption;
			static const string RelaxedMathCompilerOption;

		private:
			//This function is called by the OCLContext when attached and compiled by the context
			void SetContext(const OCLContext* const context);
			void SetProgram(cl_program program);
			void ContextDetach();
			void SetKernels();

		public:
			OCLProgram();
			virtual ~OCLProgram();

			int InstanceCount() const;

			bool ProgramSet() const;
			cl_program GetProgram() const;

			bool ContextSet() const;
			const OCLContext* const GetContext() const;

			void SetUseRelaxedMath(bool value);
			bool GetEnableUseRelaxedMath() const;

			void SetEnableMAD(bool value);
			bool GetEnableMAD() const;

			void SetEnableMatunaScript(bool value);
			bool GetEnableMatunaScript() const;

			void SetName(string name);
			string GetName() const;

			void AddDefine(string name);
			void AddDefine(string name, string value);

			void RemoveDefine(string name);

			void AddIncludePath(string includePath);
			bool IncludePathAdded(string includePath) const;
			void RemoveIncludePath(string includePath);

			void AddProgramPath(string codePath);
			void RemoveProgramPath(string codePath);

			//To move the ownership to this class, is more complex but safer!
			void AttachKernel(unique_ptr<OCLKernel> kernel);
			void DetachKernel(OCLKernel* kernel);

			void Reset();

			string GetCompilerOptions() const;
			vector<string> GetProgramCodeFiles() const;

			template <class T>
			T* GetKernel(string name) const
			{
				return dynamic_cast<T*>(kernels[name].get());
			}
		};

	} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_OCLPROGRAM_H_ */
