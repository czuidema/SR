/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    debug.hpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * cuda_debug.hpp
 *
 *  Created on: Apr 18, 2012
 *      Author: steinbrf
 */

#ifndef CUDA_DEBUG_HPP_
#define CUDA_DEBUG_HPP_

#include <string>

void saveFloatImage
(
		std::string savename,
		float *image,
		int   width,
		int   height,
		int   channels,
		float minimum = 0.0f,
		float maximum = 255.0f
);

void showFloatImage
(
		std::string showname,
		float *image,
		int   width,
		int   height,
		int   channels,
		float minimum = 0.0f,
		float maximum = 255.0f
);

void saveCudaImage
(
		std::string savename,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum = 0.0f,
		float maximum = 255.0f
);

void showCudaImage
(
		std::string savename,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum = 0.0f,
		float maximum = 255.0f
);

void saveCudaImageReciprocal
(
		std::string savename,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum = 0.0f,
		float maximum = 255.0f
);

void showCimg
(
		std::string showname,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum,
		float maximum
);

#endif /* CUDA_DEBUG_HPP_ */
