#include "RealType.h"

//<!@
#define BORDER_START_LEFT -1
#define BORDER_START_RIGHT -1
#define BORDER_START_UP -1
#define BORDER_START_DOWN -1
#define BORDER_LIMIT_LEFT -1
#define BORDER_LIMIT_RIGHT -1
#define BORDER_LIMIT_UP -1
#define BORDER_LIMIT_DOWN -1
#define BORDER_SIZE_HORIZONTAL -1
#define BORDER_SIZE_VERTICAL -1
#define INPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_UNIT_OFFSET -1
#define INPUT_UNIT_WIDTH -1
#define INPUT_UNIT_HEIGHT -1
#define INPUT_UNIT_MEMORY_WIDTH -1
//!@>

__kernel void ZeroBorderKernel(__global real_t* input)
{
	const int unitIndex = (get_global_id(0) + INPUT_UNIT_OFFSET) * INPUT_UNIT_MEMORY_ELEMENTS;

	//Adding a border in the height direction
	int tempIndex;
	const int toNextBorder = INPUT_UNIT_WIDTH + BORDER_SIZE_HORIZONTAL;
	for (int j = BORDER_LIMIT_UP + 1; j < BORDER_START_DOWN; j++)
	{
		tempIndex = INPUT_UNIT_MEMORY_WIDTH * j + unitIndex;
		for (int i = BORDER_START_LEFT; i <= BORDER_LIMIT_LEFT; i++)
		{
			input[tempIndex + i] = 0;
			input[tempIndex + i + toNextBorder] = 0;
		}
	}

	int tempIndex2;
	const int toNextBorder2 = INPUT_UNIT_HEIGHT + BORDER_SIZE_VERTICAL;
	for (int j = BORDER_START_UP; j <= BORDER_LIMIT_UP; j++)
	{
		tempIndex = INPUT_UNIT_MEMORY_WIDTH * j + unitIndex;
		tempIndex2 = tempIndex + INPUT_UNIT_MEMORY_WIDTH * toNextBorder2;
		for (int i = BORDER_START_LEFT; i <= BORDER_LIMIT_RIGHT; i++)
		{
			input[tempIndex + i] = 0;
			input[tempIndex2 + i] = 0;
		}
	}
}
