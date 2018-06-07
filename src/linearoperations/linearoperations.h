/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    linearoperations.h
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * blur_resample.h
 *
 *  Created on: Mar 15, 2012
 *      Author: steinbrf
 */

#ifndef BLUR_RESAMPLE_H_
#define BLUR_RESAMPLE_H_
#include <stdlib.h>

/*!
 * @brief Resamples an Image in X and Y separatly, without other running
 * variables than x and y
 * @param in Input image
 * @param out Output image
 * @param nx_in Input width
 * @param ny_in Input height
 * @param nx_out Output width
 * @param ny_out Output height
 * @param help Helper Array of Size nx_out*ny_in, if NULL,
 * the method will allocate it itself
 * @param scalefactor The resampled image values are scaled with this value
 */
void resampleAreaParallelizableSeparate
(
	const float *in,
	float *out,
	int   nx_in,
	int   ny_in,
	int   nx_out,
	int   ny_out,
	float *help = NULL,
	float scalefactor = 1.0f
);

/*!
 * @brief Adjoined version of the resample function,
 * with different normalization
 * @param in Input image
 * @param out Output image
 * @param nx_in Input width
 * @param ny_in Input height
 * @param nx_out Output width
 * @param ny_out Output height
 * @param help Helper Array of Size nx_out*ny_in, if NULL,
 * the method will allocate it itself
 * @param scalefactor The resampled image values are scaled with this value
 */
void resampleAreaParallelizableSeparateAdjoined
(
	const float *in,
	float *out,
	int   nx_in,
	int   ny_in,
	int   nx_out,
	int   ny_out,
	float *help = NULL,
	float scalefactor = 1.0f
);


/*!
 * @brief Bilinear backward registration (warping). If the displacement
 * vector points outside the image boundaries, the pixel in the registered
 * image is set to the provided value
 * @param f2 Input image
 * @param f2_bw Registered output image
 * @param u Horizontal component of the displacement field
 * @param v Vertical component of the displacement field
 * @param value Value to be used if displacement field points out of
 * the image boundaries
 * @param nx Width
 * @param ny Height
 * @param hx Pixel width, by which the horizontal displacement is divided
 * @param hy Pixel height, by which the vertical displacement is divided
 */
void backwardRegistrationBilinearValue
(
 const float *f2,
 float *f2_bw,
 const float *u,
 const float *v,
 float value,
 int   nx,
 int   ny,
 float hx = 1.0f,
 float hy = 1.0f
);

/*!
 * @brief Bilinear backward registration (warping). If the displacement
 * vector points outside the image boundaries, the pixel in the registered
 * image is set to the value at the corresponding point
 * in the provided function
 * @param f2 Input image
 * @param f2_bw Registered output image
 * @param u Horizontal component of the displacement field
 * @param v Vertical component of the displacement field
 * @param constant Function to be used if displacement field points out of
 * the image boundaries
 * @param nx Width
 * @param ny Height
 * @param hx Pixel width, by which the horizontal displacement is divided
 * @param hy Pixel height, by which the vertical displacement is divided
 */
void backwardRegistrationBilinearFunction
(
 const float *f2,
 float *f2_bw,
 const float *u,
 const float *v,
 const float *constant,
 int   nx,
 int   ny,
 float hx = 1.0f,
 float hy = 1.0f
);

/*!
 * @brief Adjoined Version of the bilinear backward registration function
 * @param in Input image
 * @param out Output image
 * @param u Horizontal component of the displacement field
 * @param v Vertical component of the displacement field
 * @param nx Width
 * @param ny Height
 */
void forewardRegistrationBilinear
(
	float *in,
	float *out,
	float *u,
	float *v,
	int   nx,
	int   ny
);

/*!
 * @brief Gaussian blur operator. Image values beyond the image
 * boundaries are mirrored inside the image. This way, the average grey
 * value is being preserved and the operator is self-adjoined
 * @param in Input image
 * @param out Output image
 * @param nx Width
 * @param ny Height
 * @param sigmax Standard deviation in X direction
 * @param sigmay Standard deviation in Y direction
 * @param radius Discrete kernel radius
 * @param temp Temporary helper memory. If NULL, the method will
 * allocate it itself
 * @param mask Memory for storage of the convolution kernel.
 * If NULL, the method will allocate it itself
 */
void gaussBlurSeparateMirror
(
	float *in,
	float *out,
	int   nx,
	int   ny,
	float sigmax,
	float sigmay,
	int   radius = 0,
	float *temp = NULL,
	float *mask = NULL
);


#endif
