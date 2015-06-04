/*
* MNISTSample1.cpp
*
*  Created on: Jun 4, 2015
*      Author: Mikael
*/

#include "AssetLoader.h"
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[])
{

	auto trainingImages = AssetLoader<float>::ReadTrainingImages();
	auto trainingTargets = AssetLoader<float>::ReadTrainingTargets();

	return 0;
}



