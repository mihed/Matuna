#include "RealType.h"

// Inclusive
#ifndef BORDER_START_LEFT 
#define BORDER_START_LEFT -1
#endif

// Inclusive
#ifndef BORDER_START_RIGHT 
#define BORDER_START_RIGHT -1
#endif

// Inclusive
#ifndef BORDER_START_UP 
#define BORDER_START_UP -1
#endif

// Inclusive
#ifndef BORDER_START_DOWN 
#define BORDER_START_DOWN -1
#endif

// BORDER_START_LEFT + BORDER_SIZE - 1. Inclusive
#ifndef BORDER_LIMIT_LEFT
#define BORDER_LIMIT_LEFT -1
#endif

// BORDER_START_RIGHT + BORDER_SIZE - 1. Inclusive
#ifndef BORDER_LIMIT_RIGHT
#define BORDER_LIMIT_RIGHT -1
#endif

// BORDER_START_UP + BORDER_SIZE - 1. Inclusive
#ifndef BORDER_LIMIT_UP
#define BORDER_LIMIT_UP -1
#endif

// BORDER_START_DOWN + BORDER_SIZE - 1. Inclusive
#ifndef BORDER_LIMIT_DOWN
#define BORDER_LIMIT_DOWN -1
#endif

#ifndef BORDER_SIZE_HORIZONTAL
#define BORDER_SIZE_HORIZONTAL -1
#endif

#ifndef BORDER_SIZE_VERTICAL
#define BORDER_SIZE_VERTICAL -1
#endif

#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

#ifndef INPUT_DATA_WIDTH
#define INPUT_DATA_WIDTH -1
#endif

#ifndef INPUT_DATA_HEIGHT
#define INPUT_DATA_HEIGHT -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

__kernel void ZeroBorderKernel(__global real_t* input)
{
	const int unitIndex = (get_global_id(0) + INPUT_UNIT_OFFSET) * INPUT_UNIT_ELEMENT_COUNT_INC_PADDING;

	//Adding a border in the height direction
	int tempIndex;
	const int toNextBorder = INPUT_DATA_WIDTH + BORDER_SIZE_HORIZONTAL;
	for (int j = BORDER_LIMIT_UP + 1; j < BORDER_START_DOWN; j++)
	{
		tempIndex = INPUT_STRIDE * j + unitIndex;
		for (int i = BORDER_START_LEFT; i <= BORDER_LIMIT_LEFT; i++)
		{
			input[tempIndex + i] = 0;
			input[tempIndex + i + toNextBorder] = 0;
		}
	}

	int tempIndex2;
	const int toNextBorder2 = INPUT_DATA_HEIGHT + BORDER_SIZE_VERTICAL;
	for (int j = BORDER_START_UP; j <= BORDER_LIMIT_UP; j++)
	{
		tempIndex = INPUT_STRIDE * j + unitIndex;
		tempIndex2 = tempIndex + INPUT_STRIDE * toNextBorder2;
		for (int i = BORDER_START_LEFT; i <= BORDER_LIMIT_RIGHT; i++)
		{
			input[tempIndex + i] = 0;
			input[tempIndex2 + i] = 0;
		}
	}
}