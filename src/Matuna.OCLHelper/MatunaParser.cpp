/*
* MatunaParser.cpp
*
*  Created on: Jun 9, 2015
*      Author: Mikael
*/

#include "MatunaParser.h"
#include "Matuna.Helper/FileHelper.h"
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <sstream>

namespace Matuna
{
	namespace Helper
	{

		const string MatunaParser::beginTag = "//<!@";
		const string MatunaParser::endTag = "//!@>";
		const string MatunaParser::commentTag = "//";
		const string MatunaParser::defineTag = "#define";
		const string MatunaParser::undefTag = "#undef";
		const string MatunaParser::ifdefTag = "#ifdef";
		const string MatunaParser::endifTag = "#endif";

		MatunaParser::MatunaParser()
		{

		}

		MatunaParser::~MatunaParser()
		{

		}

		unordered_map<string, string> MatunaParser::Parse(IMatunaParsable* element,  vector<string> sourcePaths)
		{
			auto defineSubstitutes = element->GetDefineSubstitutes();
			auto defines = element->GetDefines();
			unordered_map<string, string> result;
			for (auto& sourcePath : sourcePaths)
			{

				string line;
				auto sourceCode = FileHelper::GetTextFromPath(sourcePath);
				string sourceCodeCopy = sourceCode;
				size_t startPosition = 0;
				while((startPosition = sourceCodeCopy.find(beginTag, startPosition)) != string::npos)
				{
					vector<string> definesInsideTag;
					stringstream tagStream(sourceCodeCopy.substr(startPosition));
					getline(tagStream, line);
					bool foundEndTag = false;
					while(getline(tagStream, line))
					{
						if (line.find(endTag) != string::npos)
						{
							foundEndTag = true;
							break;
						}

						auto commentTagPosition = line.find(commentTag);
						if (commentTagPosition != string::npos)
						{
							string lineCopy = line;
							lineCopy.replace(commentTagPosition,
								commentTagPosition + commentTag.size(), "");

							vector<string> defineVector;
							istringstream iss(lineCopy);
							copy(istream_iterator<string>(iss),
								istream_iterator<string>(),
								back_inserter(defineVector));

							if (defineVector.size() != 2)
								throw runtime_error(
								"The file is invalid, check the define macros inside the matuna tags.");
							if (defineVector[0].compare(defineTag) != 0)
								"The file is invalid, check the define macros inside the matuna tags.";

							definesInsideTag.push_back(defineVector[1]);
						}
						else
						{
							vector<string> defineVector;
							istringstream iss(line);
							copy(istream_iterator<string>(iss),
								istream_iterator<string>(),
								back_inserter(defineVector));

							if (defineVector.size() != 2 && defineVector.size() != 3)
								throw runtime_error(
								"The file is invalid, check the define macros inside the matuna tags.");
							if (defineVector[0].compare(defineTag) != 0)
								"The file is invalid, check the define macros inside the matuna tags.";

							definesInsideTag.push_back(defineVector[1]);
						}
					}

					if (!foundEndTag)
						throw invalid_argument("The file does not contain valid begin / end tags");

					//Let us now add the undefs to the beginning of the matuna tag
					stringstream undefStream;
					for (auto undefDefine : definesInsideTag)
					{
						undefStream << endl << ifdefTag << " " << undefDefine << endl;
						undefStream << undefTag << " " << undefDefine << endl;
						undefStream << endifTag << endl;
					}


					undefStream << endl;
					sourceCode.insert(startPosition, undefStream.str());
					startPosition += beginTag.size();
				}

				stringstream inStream(sourceCode);
				stringstream outStream;
				auto substitute = defineSubstitutes[sourcePath];
				auto define = defines[sourcePath];
				int counter = 0;
				while (getline(inStream, line))
				{
					if (line.find(beginTag) != string::npos)
					{
						if (counter < 0)
							throw runtime_error(
							"The file is invalid, check the matuna tags.");
						counter++;
					}
					else if (line.find(endTag) != string::npos)
					{
						if (counter == 0)
							throw runtime_error(
							"The file is invalid, check the matuna tags.");
						counter--;
					}
					else if (counter == 0)
						outStream << line << endl;
					else
					{
						//Check whether it's a substitute or a define
						auto commentTagPosition = line.find(commentTag);
						if (commentTagPosition != string::npos)
						{
							string lineCopy = line;
							lineCopy.replace(commentTagPosition,
								commentTagPosition + commentTag.size(), "");

							vector<string> defineVector;
							istringstream iss(lineCopy);
							copy(istream_iterator<string>(iss),
								istream_iterator<string>(),
								back_inserter(defineVector));

							if (defineVector.size() != 2)
								throw runtime_error(
								"The file is invalid, check the define macros inside the matuna tags.");
							if (defineVector[0].compare(defineTag) != 0)
								"The file is invalid, check the define macros inside the matuna tags.";

							//Check whether the second word is given inside the defines
							if (define.find(defineVector[1]) != define.end())
								outStream << lineCopy << endl;
						}
						else
						{
							vector<string> defineVector;
							istringstream iss(line);
							copy(istream_iterator<string>(iss),
								istream_iterator<string>(),
								back_inserter(defineVector));

							if (defineVector.size() != 2 && defineVector.size() != 3)
								throw runtime_error(
								"The file is invalid, check the define macros inside the matuna tags.");

							defineVector.pop_back();

							if (defineVector[0].compare(defineTag) != 0)
								"The file is invalid, check the define macros inside the matuna tags.";

							auto substituteValue = substitute.find(defineVector[1]);
							if (substituteValue != substitute.end())
							{
								auto keyPair = *substituteValue;
								outStream << defineVector[0] << " " << keyPair.first << " " 
									<< keyPair.second << endl;
							}
							else
								throw exception(
								"The define could not be found in the parsable object");

						}
					}
				}

				string parsedCode = outStream.str();
				result.insert(make_pair(sourcePath, parsedCode));
			}

			return result;
		}

	} /* namespace Helper */
} /* namespace Matuna */
