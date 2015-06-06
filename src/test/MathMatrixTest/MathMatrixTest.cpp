/*
 * MatrixTest.cpp
 *
 *  Created on: May 13, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Math/Matrix.h"
#include <math.h>
#include <random>

using namespace Matuna::Math;

SCENARIO("Rotating a matrix")
{
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> distribution(2, 4);

	WHEN("Rotating 90 degrees four times")
	{
		auto test = Matrixf::RandomNormal(distribution(mt), distribution(mt));
		THEN("Then it should be equal to the original matrix")
		{
			Matrixf result = test.Rotate90().Rotate90().Rotate90().Rotate90();
			CHECK(result.RowCount() == test.RowCount());
			CHECK(result.ColumnCount() == test.ColumnCount());
			for (int i = 0; i < result.ElementCount(); i++)
				CHECK(result.Data[i] == test.Data[i]);
		}
	}

	WHEN("Rotating 90 degrees two times")
	{
		auto test = Matrixf::RandomNormal(distribution(mt), distribution(mt));
		THEN("It should be equal to rotating 180 degrees")
		{
			Matrixf result1 = test.Rotate90().Rotate90();
			Matrixf result2 = test.Rotate180();
			CHECK(result1.RowCount() == result2.RowCount());
			CHECK(result1.ColumnCount() == result2.ColumnCount());
			CHECK(result1.ElementCount() == result2.ElementCount());
			for (int i = 0; i < result1.ElementCount(); i++)
				CHECK(result1.Data[i] == result2.Data[i]);
		}
	}

	WHEN("Rotating 90 degrees three times")
	{
		auto test = Matrixf::RandomNormal(distribution(mt), distribution(mt));
		THEN("Then it should be equal to rotating 270 degrees")
		{
			Matrixf result1 = test.Rotate90().Rotate90().Rotate90();
			Matrixf result2 = test.Rotate270();
			CHECK(result1.RowCount() == result2.RowCount());
			CHECK(result1.ColumnCount() == result2.ColumnCount());
			CHECK(result1.ElementCount() == result2.ElementCount());
			for (int i = 0; i < result1.ElementCount(); i++)
				CHECK(result1.Data[i] == result2.Data[i]);
		}
	}
}

SCENARIO("Multiplying a matrix")
{
	GIVEN("5x5 matrix and 5x3 matrix")
	{
		Matrix<double> test(5, 5);
		test.At(0, 0) = 2.9387;		test.At(0, 1) = -8.0950;	test.At(0, 2) = 13.7030;	test.At(0, 3) = 3.1286;		test.At(0, 4) = 10.9327;
		test.At(1, 0) = -7.8728;	test.At(1, 1) = -29.4428;	test.At(1, 2) = -17.1152;	test.At(1, 3) = -8.6488;	test.At(1, 4) = 11.0927;
		test.At(2, 0) = 8.8840;		test.At(2, 1) = 14.3838;	test.At(2, 2) = -1.0224;	test.At(2, 3) = -0.3005;	test.At(2, 4) = -8.6365;
		test.At(3, 0) = -11.4707;	test.At(3, 1) = 3.2519;		test.At(3, 2) = -2.4145;	test.At(3, 3) = -1.6488;	test.At(3, 4) = 0.7736;
		test.At(4, 0) = -10.6887;	test.At(4, 1) = -7.5493;	test.At(4, 2) = 3.1921;		test.At(4, 3) = 6.2771;		test.At(4, 4) = -12.1412;

		Matrix<double> test2(5, 3);
		test2.At(0, 0) = -11.1350;	test2.At(0, 1) = -2.2558;	test2.At(0, 2) = 11.0061;
		test2.At(1, 0) = -0.0685;	test2.At(1, 1) = 11.1736;	test2.At(1, 2) = 15.4421;
		test2.At(2, 0) = 15.3263;	test2.At(2, 1) = -10.8906;	test2.At(2, 2) = 0.8593;
		test2.At(3, 0) = -7.6967;	test2.At(3, 1) = 0.3256;	test2.At(3, 2) = -14.9159;
		test2.At(4, 0) = 3.7138;	test2.At(4, 1) = 5.5253;	test2.At(4, 2) = -7.4230;

		Matrix<double> resultTest(5, 3);
		resultTest.At(0, 0) = 194.3699;		resultTest.At(0, 1) = -184.8890;	resultTest.At(0, 2) = -208.7038;
		resultTest.At(1, 0) = -64.8686;		resultTest.At(1, 1) = -66.3519;		resultTest.At(1, 2) = -509.3532;
		resultTest.At(2, 0) = -145.3394;	resultTest.At(2, 1) = 103.9954;		resultTest.At(2, 2) = 387.6071;
		resultTest.At(3, 0) = 106.0618;		resultTest.At(3, 1) = 92.2441;		resultTest.At(3, 2) = -59.2554;
		resultTest.At(4, 0) = 75.0562;		resultTest.At(4, 1) = -160.0436;	resultTest.At(4, 2) = -234.9790;

		WHEN("Multiplying the matrices")
		{
			INFO("Calculating with the multiplication operators")
				auto calculatedResult = test * test2;
			THEN("The result must be equal to the benchmark result")
			{
				for (int i = 0; i < 5; i++)
					for (int j = 0; j < 3; j++)
					{
						auto difference = abs((resultTest.Data[i * 3 + j] - calculatedResult.Data[i * 3 + j]) / resultTest.Data[i * 3 + j]);
						CHECK(difference < 1E-4);
					}
			}
		}
	}
}

SCENARIO("Calculating the L2 norm of a vector")
{
	auto randomMatrix = Matrix<float>::RandomNormal(5, 2);
	auto norm = randomMatrix.Norm2();
	auto data = randomMatrix.Data;

	float sum = 0;
	for (int i = 0; i < randomMatrix.ElementCount(); i++)
		sum += data[i] * data[i];

	sum = sqrt(sum);
	auto absDiffernece = abs(sum - norm);
	CHECK(absDiffernece < 1E-7);
}

SCENARIO("Transposing a matrix")
{
	random_device device;
	mt19937 mersienne(device());
	uniform_int_distribution<int> distribution(1, 100);
	for (int i = 0; i < 10; i++)
	{
		auto test1 = Matrix<double>::RandomNormal(distribution(mersienne), distribution(mersienne));
		auto test2 = test1.Transpose().Transpose();
		for (int i = 0; i < test1.ElementCount(); i++)
			CHECK(test1.Data[i] == test2.Data[i]);
	}
}

SCENARIO("Subtracting a matrix")
{
	random_device device;
	mt19937 mersienne(device());
	uniform_int_distribution<int> distribution(1, 100);

	WHEN("Subtracting with a matrix")
	{
		for (int i = 0; i < 10; i++)
		{
			auto test1 = Matrix<double>::RandomNormal(distribution(mersienne), distribution(mersienne));
			auto test2 = Matrix<double>::RandomNormal(test1.RowCount(), test1.ColumnCount());

			auto result = test1 - test2;
			for (int i = 0; i < test1.ElementCount(); i++)
				CHECK(result.Data[i] == (test1.Data[i] - test2.Data[i]));
		}
	}
	WHEN("Subtracting with a scalar")
	{
		for (int i = 0; i < 10; i++)
		{
			auto test1 = Matrix<double>::RandomNormal(distribution(mersienne), distribution(mersienne));
			double scalar = distribution(mersienne);
			auto result = test1 - scalar;

			for (int i = 0; i < test1.ElementCount(); i++)
				CHECK(result.Data[i] == (test1.Data[i] - scalar));
		}
	}
}

SCENARIO("Adding a matrix")
{
	random_device device;
	mt19937 mersienne(device());
	uniform_int_distribution<int> distribution(1, 100);

	WHEN("Adding with a matrix")
	{
		for (int i = 0; i < 10; i++)
		{
			auto test1 = Matrix<double>::RandomNormal(distribution(mersienne), distribution(mersienne));
			auto test2 = Matrix<double>::RandomNormal(test1.RowCount(), test1.ColumnCount());

			auto result = test1 + test2;
			for (int i = 0; i < test1.ElementCount(); i++)
				CHECK(result.Data[i] == (test1.Data[i] + test2.Data[i]));
		}
	}
	WHEN("Adding with a scalar")
	{
		for (int i = 0; i < 10; i++)
		{
			auto test1 = Matrix<double>::RandomNormal(distribution(mersienne), distribution(mersienne));
			double scalar = distribution(mersienne);
			auto result = test1 + scalar;

			for (int i = 0; i < test1.ElementCount(); i++)
				CHECK(result.Data[i] == (test1.Data[i] + scalar));
		}
	}
}

SCENARIO("Element-wise multiplication")
{
	random_device device;
	mt19937 mersienne(device());
	uniform_int_distribution<int> distribution(1, 100);

	WHEN("Multiplying with a matrix")
	{
		for (int i = 0; i < 10; i++)
		{
			auto test1 = Matrix<double>::RandomNormal(distribution(mersienne), distribution(mersienne));
			auto test2 = Matrix<double>::RandomNormal(test1.RowCount(), test1.ColumnCount());

			auto result = test1 % test2;
			for (int i = 0; i < test1.ElementCount(); i++)
				CHECK(result.Data[i] == (test1.Data[i] * test2.Data[i]));
		}
	}
	WHEN("Multiplying with a scalar")
	{
		for (int i = 0; i < 10; i++)
		{
			auto test1 = Matrix<double>::RandomNormal(distribution(mersienne), distribution(mersienne));
			double scalar = distribution(mersienne);
			auto result = scalar * test1;

			for (int i = 0; i < test1.ElementCount(); i++)
				CHECK(result.Data[i] == scalar * test1.Data[i]);
		}
	}
}