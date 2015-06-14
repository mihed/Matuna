/*
* OCLProgram.cpp
*
*  Created on: Jun 9, 2015
*      Author: Mikael
*/

#include "OCLProgram.h"
#include "OCLUtility.h"
#include "MatunaParser.h"

#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

#include <stdexcept>

namespace Matuna {
	namespace Helper {

		string OCLProgram::DefaultSourceLocation = Path::Combine(Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels/");
		int OCLProgram::instanceCounter = 0;
		const string OCLProgram::MADCompilerOption = " -cl-mad-enable ";
		const string OCLProgram::RelaxedMathCompilerOption = " -cl-fast-relaxed-math ";
		const string OCLProgram::WarningToErrorOption = "-Werror";

		OCLProgram::OCLProgram() 
		{
			enableMAD = false;
			enableMatunaScript = true;
			useRelaxedMath = false;
			context = nullptr;
			program = nullptr;

			instanceCounter++;
		}

		OCLProgram::~OCLProgram() 
		{
			//First, clear the unique_ptr to the kernels and then the program.
			kernels.clear();

			if (ProgramSet())
				CheckOCLError(clReleaseProgram(program), "Could not release the program");
		}

		int OCLProgram::InstanceCount() const
		{
			return instanceCounter;
		}

		bool OCLProgram::ProgramSet() const
		{
			return program == nullptr ? false : true;
		}

		cl_program OCLProgram::GetProgram() const
		{
			return program;
		}

		bool OCLProgram::ContextSet() const
		{
			return context == nullptr ? false : true;
		}

		const OCLContext* const OCLProgram::GetContext() const
		{
			return context;
		}

		void OCLProgram::SetUseRelaxedMath(bool value)
		{
			this->useRelaxedMath = value;
		}

		bool OCLProgram::GetEnableUseRelaxedMath() const
		{
			return useRelaxedMath;
		}

		void OCLProgram::SetEnableMAD(bool value)
		{
			this->enableMAD = value;
		}

		bool OCLProgram::GetEnableMAD() const
		{
			return enableMAD;
		}

		void OCLProgram::SetEnableMatunaScript(bool value)
		{
			this->enableMatunaScript = value;
		}

		bool OCLProgram::GetEnableMatunaScript() const
		{
			return enableMatunaScript;
		}

		void OCLProgram::SetName(string name)
		{
			this->name = name;
		}

		string OCLProgram::GetName() const
		{
			return name;
		}

		void OCLProgram::AddDefine(string name)
		{
			if (defines.find(name) == defines.end())
				defines.insert(name);
		}

		void OCLProgram::AddDefine(string name, string value)
		{
			if (definesWithValues.find(name) == definesWithValues.end())
				definesWithValues.insert(make_pair(name, value));
		}

		void OCLProgram::RemoveDefine(string name)
		{
			if (defines.find(name) != defines.end())
				defines.erase(name);

			if (definesWithValues.find(name) != definesWithValues.end())
				definesWithValues.erase(name);
		}

		void OCLProgram::AddIncludePath(string includePath)
		{
			if (includePaths.find(includePath) == includePaths.end())
				includePaths.insert(includePath);
		}

		bool OCLProgram::IncludePathAdded(string includePath) const
		{
			return includePaths.find(includePath) == includePaths.end() ? false : true;
		}

		void OCLProgram::RemoveIncludePath(string includePath)
		{
			if (includePaths.find(includePath) != includePaths.end())
				includePaths.erase(includePath);
		}

		void OCLProgram::AddProgramPath(string codePath)
		{
			if (pathAndCodeFiles.find(codePath) == pathAndCodeFiles.end())
				pathAndCodeFiles.insert(make_pair(codePath, FileHelper::GetTextFromPath(codePath)));
		}

		void OCLProgram::RemoveProgramPath(string codePath)
		{
			if (pathAndCodeFiles.find(codePath) != pathAndCodeFiles.end())
				pathAndCodeFiles.erase(codePath);
		}

		void OCLProgram::AttachKernel(unique_ptr<OCLKernel> kernel)
		{
			string kernelName = kernel->Name();

			if (kernels.find(kernelName) != kernels.end())
				throw runtime_error("The kernel is already attached");

			OCLSourceKernel* sourceKernelPointer = dynamic_cast<OCLSourceKernel*>(kernel.get()); 
			if (sourceKernelPointer)
			{
				auto kernelIncludePaths = sourceKernelPointer->GetIncludePaths();
				auto parsable = dynamic_cast<IMatunaParsable*>(sourceKernelPointer);
				if (parsable && enableMatunaScript)
				{
					MatunaParser parser;
					auto pathCodeMap = parser.Parse(parsable, sourceKernelPointer->GetSourcePaths());
					//TODO: In this case we can actually miss matuna parsed files and files that have not been matuna parsed!
					for (auto& pathCodePair : pathCodeMap)
						if (pathAndCodeFiles.find(pathCodePair.first) == pathAndCodeFiles.end())
							pathAndCodeFiles.insert(make_pair(pathCodePair.first, pathCodePair.second));
				}
				else
				{
					auto kernelSourcePaths = sourceKernelPointer->GetSourcePaths();
					for (auto path : kernelSourcePaths)
						AddProgramPath(path);
				}

				for (auto includePath : kernelIncludePaths)
					AddIncludePath(includePath);
			}

			kernel->SetProgram(this);
			kernels.insert(make_pair(kernelName, move(kernel)));
		}

		void OCLProgram::DetachKernel(OCLKernel* kernel)
		{
			kernel->ProgramDetach();
			kernels.erase(kernel->Name());
		}

		string OCLProgram::GetCompilerOptions() const
		{
			stringstream result;

			result << WarningToErrorOption << " ";

			if (enableMAD)
				result << MADCompilerOption;
			if(useRelaxedMath)
				result << RelaxedMathCompilerOption;

			for (auto& define : defines)
				result << " -D" << define;

			for (auto& defineWithValue : definesWithValues)
				result << " -D" << defineWithValue.first << "=" << defineWithValue.second;

			for(auto& includePath : includePaths)
				result << " -I" << includePath;

			return result.str();
		}

		vector<string> OCLProgram::GetProgramCodeFiles() const
		{
			vector<string> result;
			for (auto& codePair : pathAndCodeFiles)
				result.push_back(codePair.second);

			return result;
		}

		void OCLProgram::Reset()
		{
			enableMAD = false;
			enableMatunaScript = true;
			useRelaxedMath = false;
			kernels.clear();
			includePaths.clear();
			pathAndCodeFiles.clear();
			context = nullptr;
			name = "";
			if (ProgramSet())
			{
				CheckOCLError(clReleaseProgram(program), "Could not release the program");
				program = nullptr;
			}
		}

		void OCLProgram::ContextDetach()
		{

			//TODO: Observe that if we added source kernels, their files are still around!
			if (!ProgramSet())
				throw runtime_error("The context cannot detach itself if it has not been attached");

			for(auto& kernelPair : kernels)
				kernelPair.second->ProgramDetach();

			kernels.clear();

			CheckOCLError(clReleaseProgram(program), "Could not release the program");
			program = nullptr;
			context = nullptr;
		}

		void OCLProgram::SetContext(const OCLContext* const context)
		{
			if (ContextSet())
				throw runtime_error("The context has been set, please reset in order to re-use it.");

			this->context = context;
		}

		void OCLProgram::SetProgram(cl_program program)
		{
			if (ProgramSet())
				throw runtime_error("The program has been set, please reset in order to re-use it.");

			this->program = program;

			SetKernels();
		}

		void OCLProgram::SetKernels()
		{
			for(auto& kernelPair : kernels)
			{
				cl_int errorCode;
				cl_kernel kernel = clCreateKernel(program, kernelPair.second->Name().c_str(), &errorCode);
				CheckOCLError(errorCode, "The kernel could not be created, please check that the name is correct");
				auto kernelPointer = kernelPair.second.get();
				kernelPointer->SetKernel(kernel);

				if (kernelPointer->GetProgram() != this)
					throw runtime_error("The program of the kernel does not correspond to the actual attached program.");
				if (!kernelPointer->KernelSet())
					throw runtime_error("The kernel has not been set properly.");
			}
		}

	} /* namespace Helper */
} /* namespace Matuna */
