// Copyright 2009-2015 Intel Corporation.
//
// The source code, information and material ("Material") contained herein is 
// owned by Intel Corporation or its suppliers or licensors, and title to such 
// Material remains with Intel Corporation or its suppliers or licensors. 
// The Material contains proprietary information of Intel or its suppliers and 
// licensors. The Material is protected by worldwide copyright laws and treaty 
// provisions. No part of the Material may be used, copied, reproduced, modified, 
// published, uploaded, posted, transmitted, distributed or disclosed in any way 
// without Intel's prior express written permission. No license under any patent, 
// copyright or other intellectual property rights in the Material is granted to 
// or conferred upon you, either expressly, by implication, inducement, estoppel 
// or otherwise. Any license under such intellectual property rights must be 
// express and approved by Intel in writing.
//
// *OpenCL(TM) - OpenCL and the OpenCL logo are trademarks of Apple Inc. used by 
// permission by Khronos.
//
// *Third Party trademarks are the property of their respective owners.
//
// Intel and the Intel logo are trademarks of Intel Corporation in the U.S. and/or 
// other countries.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter this 
// notice or any other notice embedded in Materials by Intel or Intel's suppliers 
// or licensors in any way.




#include "stdafx.h"



// random shuffling of 0-255  -  this code could be modified to allow user-provided permute table
const	unsigned int	default_perm[256] = {
	151, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 99, 37,
	8, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 160, 137, 
	35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
	134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
	55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73,
	18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,
	250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59,
	189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221,
	43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178,
	97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51,
	107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50,
	138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66,
	140, 36, 103, 30, 227, 47, 16, 58, 69, 17, 209, 76, 132, 187, 45, 127,
	197, 62, 94, 252, 153, 101, 155, 167, 219, 182, 3, 64, 52, 217, 215, 61,
	168, 68, 175, 74, 185, 112, 104, 218, 165, 246, 4, 150, 208, 254, 142, 71,
	230, 220, 105, 92, 145, 235, 249, 14, 41, 239, 156, 180, 226, 89, 203, 117
};

// 16 normalized pair vectors uniform distribution and off-axes.  4 corners randomly chosen gives 2^16 combinations
const	float	grads2d[16][2] = {
	{ -0.195090322f,-0.98078528f },
	{ -0.555570233f,-0.831469612f },
	{ -0.831469612f,-0.555570233f },
	{ -0.98078528f,-0.195090322f },
	{ -0.98078528f,0.195090322f },
	{ -0.831469612f,0.555570233f },
	{ -0.555570233f,0.831469612f },
	{ -0.195090322f,0.98078528f },
	{ 0.195090322f,0.98078528f },
	{ 0.555570233f,0.831469612f },
	{ 0.831469612f,0.555570233f },
	{ 0.98078528f,0.195090322f },
	{ 0.98078528f,-0.195090322f },
	{ 0.831469612f,-0.555570233f },
	{ 0.555570233f,-0.831469612f },
	{ 0.195090322f,-0.98078528f }
};

const	int	grads3d[16][4] = {
	{ 1, 1, 0, 0 }, { -1, 1, 0, 0 }, { 1, -1, 0, 0 }, { -1, -1, 0, 0 },
	{ 1, 0, 1, 0 }, { -1, 0, 1, 0 }, { 1, 0, -1, 0 }, { -1, 0, -1, 0 },
	{ 0, 1, 1, 0 }, { 0, -1, 1, 0 }, { 0, 1, -1, 0 }, { 0, -1, -1, 0 },
	{ 1, 1, 0, 0 }, { -1, 1, 0, 0 }, { 0, -1, 1, 0 }, { 0, -1, -1, 0 }
};



// Parallel-usable pseudo-random number generators
// Usage: parameters should be derived from work item ID(s)
// Need to iterate to get OK 2d results

#if 1

// Wang Hash based RNG
//  Has at least 20 separate cycles, shortest cycle is < 7500 long.  
//  But it yields random looking 2D noise when fed OpenCL work item IDs, 
//  The short cycle should only be hit for one work item in about 500K.
unsigned int ParallelRNG( unsigned int x )
{
	unsigned int value = x;

	value = (value ^ 61) ^ (value >> 16);
	value *= 9;
	value = value ^ (value << 4);
	value *= 0x27d4eb2d;
	value = value ^ (value >> 15);

	return value;
}

#else

// Unix OS RNG - fast, single cycle of 2^32 numbers, 
//    but not very random looking when used with OpenCL work item IDs.
unsigned int ParallelRNG( unsigned int x )
{
	unsigned int value = x;

    value = 1103515245 * value + 12345;

	return value;
}

#endif


// 2D random pattern, or deterministically selected 1D patterns by varying 'y' 
unsigned int ParallelRNG2(unsigned int x, unsigned int y)
{
	unsigned int value = ParallelRNG(x);

	value = ParallelRNG(y ^ value);

	return value;
}


// 3D random patterns, or deterministically selected 2D patterns by varying "z"
unsigned int ParallelRNG3(unsigned int x, unsigned int y, unsigned int z)
{
	unsigned int value = ParallelRNG2(x, y);

	value = ParallelRNG(z ^ value);

	return value;
}



float
weight_poly3(float weight)
{
	return	weight * weight * (3 - weight * 2);		// Perlin's original ease curve function
}


float
weight_poly5(float weight)
{
	return	weight * weight * weight * (weight * (weight * 6 - 15) + 10);		// Perlin's improved ease curve function
}


#define	WEIGHT(w)	weight_poly5(w)



// map 0-255  to -1.0 up to but excluding 1.0 
#define	NORM256(ff)	 ((1.0f / 128.0f) * ((ff) - 128.0f))


// bilinear interpolation
float	interp( float weight,  float  valA, float valB)
{ 
	return	valA + weight * (valB - valA);
}





float
hash_grad_dot2(unsigned int hash, float x, float y)					// 2d gradient look up and dot product with vector
{
	unsigned int indx = hash & 0x0f;

	return	(x * grads2d[indx][0]) + (y * grads2d[indx][1] );

}



//	This calculates the Improved Perlin noise function once for 2d using default permutation table 
//  Returned result should be between -1.0 and 1.0
float
noise_2d(float x, float y)
{
	float	X = floor(x);	// lower grid coordinates
	float	Y = floor(y);

	float	vx, vy;
	vx = x - X;		// vector from lower grid coordinates
	vy = y - Y;
	float	vX, vY;
	vX = vx - 1.0f;
	vY = vy - 1.0f;


	int	ux = (int)(X);
	int	uy = (int)(Y);
	int	uX = ux + 1;
	int	uY = uy + 1;

	unsigned int 	px = default_perm[ux & 0x0FF];				// generate permutation grads
	unsigned int	pX = default_perm[uX & 0x0FF];

	unsigned int	pxy = default_perm[(px + uy) & 0x0FF];
	unsigned int	pXy = default_perm[(pX + uy) & 0x0FF];
	unsigned int	pxY = default_perm[(px + uY) & 0x0FF];
	unsigned int	pXY = default_perm[(pX + uY) & 0x0FF];


	float	gxy = hash_grad_dot2(pxy, vx, vy);
	float	gXy = hash_grad_dot2(pXy, vX, vy);
	float	gxY = hash_grad_dot2(pxY, vx, vY);
	float	gXY = hash_grad_dot2(pXY, vX, vY);

	float	wx = WEIGHT(vx);
	float	wy = WEIGHT(vy);

	return  interp(wy, interp(wx, gxy, gXy), interp(wx, gxY, gXY));
}




float
hash_grad_dot3(unsigned int hash, float x, float y, float z)					// 3d gradient look up and dot product with vector
{
	unsigned int indx = hash & 0x0f;


	return	(x * grads3d[indx][0]) + (y * grads3d[indx][1]) + (z * grads3d[indx][2]);
}


//	This calculates the Improved Perlin noise function once for 3d using default permutation table 
//  Returned result should be between -1.0 and 1.0
float	
noise_3d(float x, float y, float z)
{
	float	X = floor(x);	// lower grid coordinates
	float	Y = floor(y);
	float	Z = floor(z);

	float	vx, vy, vz;
	vx = x - X;		// vector from lower grid coordinates
	vy = y - Y;
	vz = z - Z;

	float	vX, vY, vZ;
	vX = vx - 1.0f;
	vY = vy - 1.0f;
	vZ = vz - 1.0f;


    // 3D Grid coordinates adjacent to samplep point
	int	ux = (int)(X);
	int	uy = (int)(Y);
	int	uz = (int)(Z);
	unsigned int	uX = ux + 1;
	unsigned int	uY = uy + 1;
	unsigned int	uZ = uz + 1;


    // generate permutation grads
	unsigned int	px = default_perm[ux & 0x0FF];
	unsigned int	pX = default_perm[uX & 0x0FF];

	unsigned int	pxy = default_perm[(px + uy) & 0x0FF];
	unsigned int	pXy = default_perm[(pX + uy) & 0x0FF];
	unsigned int	pxY = default_perm[(px + uY) & 0x0FF];
	unsigned int	pXY = default_perm[(pX + uY) & 0x0FF];

	unsigned int	pxyz = default_perm[(pxy + uz) & 0x0FF];
	unsigned int	pXyz = default_perm[(pXy + uz) & 0x0FF];
	unsigned int	pxYz = default_perm[(pxY + uz) & 0x0FF];
	unsigned int	pXYz = default_perm[(pXY + uz) & 0x0FF];
	unsigned int	pxyZ = default_perm[(pxy + uZ) & 0x0FF];
	unsigned int	pXyZ = default_perm[(pXy + uZ) & 0x0FF];
	unsigned int	pxYZ = default_perm[(pxY + uZ) & 0x0FF];
	unsigned int	pXYZ = default_perm[(pXY + uZ) & 0x0FF];


    // select values for 8 adjacent grid points, dot product with vector to fractional position
	float	gxyz = hash_grad_dot3(pxyz, vx, vy, vz);        
	float	gXyz = hash_grad_dot3(pXyz, vX, vy, vz);
	float	gxYz = hash_grad_dot3(pxYz, vx, vY, vz);
	float	gXYz = hash_grad_dot3(pXYz, vX, vY, vz);
	float	gxyZ = hash_grad_dot3(pxyZ, vx, vy, vZ);
	float	gXyZ = hash_grad_dot3(pXyZ, vX, vy, vZ);
	float	gxYZ = hash_grad_dot3(pxYZ, vx, vY, vZ);
	float	gXYZ = hash_grad_dot3(pXYZ, vX, vY, vZ);

	float	wx = WEIGHT(vx);
	float	wy = WEIGHT(vy);
	float	wz = WEIGHT(vz);

        // interpolate to a single value
	float   result =  
                    interp( wz,
		                    interp( wy,
			                        interp(wx, gxyz, gXyz),
			                        interp(wx, gxYz, gXYz)
			                    ),
		                    interp( wy,
			                        interp(wx, gxyZ, gXyZ),
			                        interp(wx, gxYZ, gXYZ)
			                    )
		                    );

    return result;

}



// Generate "cloud" pattern values using Perlin Noise
float	cloud(float fx, float fy, float fz, float size)
{
	float	value = 0.0f;

	while (size >= 1.0f)
	{
		value += size * noise_3d(fx, fy, fz);

        size *= 0.5f;
		fx *= 2.0f;
		fy *= 2.0f;
	}
	return value;
}




// map -1.0 - 1.0 onto 0-255
float  map256(float v)
{
	return ((127.5f * v) + 127.5f);
}



// Demo using "noise3d" to generate "clouds" 
void CloudTest(float* outputArray, int width, int height, float slice)
{
	float	invWidth, invHeight;

	invWidth = 1.0f / width;
	invHeight = 1.0f / height;


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			unsigned int offset = 4 * (x + y *  width);

			float	fx = 2.0f * (float)x * invWidth;
			float	fy = 2.0f * (float)y * invHeight;
			float	fz = slice;

			float	size = (float)width;
			float	value = 0.0f;

			value = cloud(fx, fy, fz, size);


			value *= (float)invWidth;

			value = map256(value);

			float	RG = 255.0f - (value - 25.0f);	// biasing toward white, as value rarely gets down to 0.0
			float	B = 255.0f;                     // Sky is otherwise blue

            RG = (RG < 0.0f) ? 0.0f : RG;

			outputArray[offset + 0] = RG;
			outputArray[offset + 1] = RG;
			outputArray[offset + 2] = B;
			outputArray[offset + 3] = 255.0f;

		}
	}
}



// Generate a single 2D Perlin noise map  
//  x and y parameters should vary over 0.0 and 1.0
//  They will be scaled to generate a 128 x 72 grid spread over the width/height of outputArray.
//  User code can use any grid size, a larger grid yielding more varied noise, less continuity.
void Noise2dTest ( float* outputArray, int width, int height, float slice )
{
	float	invWidth, invHeight;

	invWidth = 1.0f / width;
	invHeight = 1.0f / height;


    // Loop to fill in a 2D buffer with grayscale values
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int offset = 4 * (x + y *  width);

			float	fx = 128 * (float)x * invWidth;
			float	fy = 72 * (float)y * invHeight;

			float	value;

			value = noise_2d(fx, fy);

			value = map256(value);

			outputArray[offset + 0] = value;
			outputArray[offset + 1] = value;
			outputArray[offset + 2] = value;
			outputArray[offset + 3] = 255.0f;

		}
	}
}



// Generate a 2D Perlin noise map, varied by "slice" parameter.  
//  x and y parameters should vary over 0.0 and 1.0
//  They will be scaled to generate a 128 x 72 grid spread over the width/height of outputArray.
//  User code can use any grid size, a larger grid yielding more varied noise, less continuity.
void Noise3dTest(float* outputArray, int width, int height, float slice)
{
	float	invWidth, invHeight;

	invWidth = 1.0f / width;
	invHeight = 1.0f / height;


    // Loop to fill in a 2D buffer with grayscale values
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int offset = 4 * (x + y *  width);

			float	fx = 128 * (float)x * invWidth;
			float	fy = 72 * (float)y * invHeight;
			float	fz = slice;

			float	value;

			value = noise_3d(fx, fy, fz);

			value = map256(value);

			outputArray[offset + 0] = value;
			outputArray[offset + 1] = value;
			outputArray[offset + 2] = value;
			outputArray[offset + 3] = 255.0f;

		}
	}
}



// Fill a 2D outputArray buffer with reproducible random noise, 
// The  slice parameter selects different random 2D patterns.
void RandomTest(float* outputArray, int width, int height, float slice)
{
    // x and y correlate to the OpenCL kernel function values generated from global IDs
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int offset = 4 * (x + y *  width);


            // Random integer generator: slice needs to vary by integer amounts to change noise pattern
			unsigned int	value = ParallelRNG3(x, y, (int)slice);

            // only using low bits - but higher bits should be 'random' also
			value &= 0x0ff;

			outputArray[offset + 0] = (float)value;
			outputArray[offset + 1] = (float)value;
			outputArray[offset + 2] = (float)value;
			outputArray[offset + 3] = 255.0f;

		}
	}
}



