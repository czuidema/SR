/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    linearoperations.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * linearoperations.cuh
 *
 *  Created on: Aug 3, 2012
 *      Author: steinbrf
 */

#ifndef LINEAROPERATIONS_CUH
#define LINEAROPERATIONS_CUH

/*!
 * @brief Bilinear backward registration (warping). If the displacement
 * vector points outside the image boundaries, the pixel in the registered
 * image is set to the provided value. The input image is read from texture
 * @param in_g Input image
 * @param flow1_g Horizontal component of the displacement field
 * @param flow2_g Vertical component of the displacement field
 * @param out_g Registered output image
 * @param value Value to be used if displacement field points out of
 * the image boundaries
 * @param nx Output width
 * @param ny Output height
 * @param pitchf1_in Pitch of the input image
 * @param pitchf1_out Pitch of the output image and flows
 * @param hx Pixel width, by which the horizontal displacement is divided
 * @param hy Pixel height, by which the vertical displacement is divided
 */
void backwardRegistrationBilinearValueTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		float value,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx = 1.0f,
		float hy = 1.0f
);

/*!
 * @brief Bilinear backward registration (warping). If the displacement
 * vector points outside the image boundaries, the pixel in the registered
 * image is set to the value at the corresponding point
 * in the provided function. The input image is read from texture
 * @param in_g Input image
 * @param flow1_g Horizontal component of the displacement field
 * @param flow2_g Vertical component of the displacement field
 * @param out_g Registered output image
 * @param constant_g Function to be used if displacement field points out of
 * the image boundaries
 * @param nx Output width
 * @param ny Output height
 * @param pitchf1_in Pitch of the input image
 * @param pitchf1_out Pitch of the output image and flows
 * @param hx Pixel width, by which the horizontal displacement is divided
 * @param hy Pixel height, by which the vertical displacement is divided
 */
void backwardRegistrationBilinearFunctionTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx = 1.0f,
		float hy = 1.0f
);

/*!
 * @brief Bilinear backward registration (warping). If the displacement
 * vector points outside the image boundaries, the pixel in the registered
 * image is set to the value at the corresponding point
 * in the provided function. The input image is read from global memory
 * @param in_g Input image
 * @param flow1_g Horizontal component of the displacement field
 * @param flow2_g Vertical component of the displacement field
 * @param out_g Registered output image
 * @param constant_g Function to be used if displacement field points out of
 * the image boundaries
 * @param nx Output width
 * @param ny Output height
 * @param pitchf1_in Pitch of the input image
 * @param pitchf1_out Pitch of the output image and flows
 * @param hx Pixel width, by which the horizontal displacement is divided
 * @param hy Pixel height, by which the vertical displacement is divided
 */
void backwardRegistrationBilinearFunctionGlobal
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
);

/*!
 * @brief Adjoined Version of the bilinear backward registration function
 * using atomics
 * @param flow1_g Horizontal component of the displacement field
 * @param flow2_g Vertical component of the displacement field
 * @param in_g Input image
 * @param out_g Output image
 * @param nx Width
 * @param ny Height
 * @param pitchf1 Pitch
 */
void forewardRegistrationBilinearAtomic
(
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
	  float       *out_g,
		int         nx,
		int         ny,
		int         pitchf1
);

/*!
 * @brief Gaussian blur operator. Image values beyond the image
 * boundaries are mirrored inside the image. This way, the average grey
 * value is being preserved and the operator is self-adjoined
 * @param in_g Input image
 * @param out_g Output image
 * @param nx Width
 * @param ny Height
 * @param pitchf1 Pitch
 * @param sigmax Standard deviation in X direction
 * @param sigmay Standard deviation in Y direction
 * @param radius Discrete kernel radius
 * @param temp_g Temporary helper memory. If NULL, the method will
 * allocate it itself
 * @param mask Memory for storage of the convolution kernel.
 * If NULL, the method will allocate it itself
 */
void gaussBlurSeparateMirrorGpu
(
		float *in_g,
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float sigmax,
		float sigmay,
		int   radius,
		float *temp_g,
		float *mask
);

/*!
 * @brief Resamples an Image in X and Y separatly, without other running
 * variables than x and y
 * @param in_g Input image
 * @param out_g Output Image
 * @param nx_in Input width
 * @param ny_in Input height
 * @param pitchf1_in Input pitch
 * @param nx_out Output width
 * @param ny_out Output height
 * @param pitchf1_out Output pitch
 * @param help_g Helper Array of Size nx_out*ny_in, if NULL,
 * the method will allocate it itself
 * @param scalefactor The resampled image values are scaled with this value
 */
void resampleAreaParallelSeparate
(
		const float *in_g,
		float *out_g,
		int   nx_in,
		int   ny_in,
		int   pitchf1_in,
		int   nx_out,
		int   ny_out,
		int   pitchf1_out,
		float *help_g = NULL,
		float scalefactor = 1.0f
);

/*!
 * @brief Adjoined version of the resample function,
 * with different normalization
 * @param in_g Input image
 * @param out_g Output image
 * @param nx_in Input width
 * @param ny_in Input height
 * @param pitchf1_in Input pitch
 * @param nx_out Output width
 * @param ny_out Output height
 * @param pitchf1_out Output pitch
 * @param help_g Helper Array of Size nx_out*ny_in, if NULL,
 * the method will allocate it itself
 * @param scalefactor The resampled image values are scaled with this value
 */
void resampleAreaParallelSeparateAdjoined
(
		const float *in_g,
		float *out_g,
		int   nx_in,
		int   ny_in,
		int   pitchf1_in,
		int   nx_out,
		int   ny_out,
		int   pitchf1_out,
		float *help_g = NULL,
		float scalefactor = 1.0f
);

#ifdef NVCC
/*!
 * @brief Adds the increment to the accumulator
 * @param increment_g Increment
 * @param accumulator_g Accumulator
 * @param nx Width
 * @param ny Height
 * @param pitchf1 Pitch
 */
__global__ void addKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
);

/*!
 * @brief Subtracts the increment from the accumulator
 * @param increment_g Increment
 * @param accumulator_g Accumulator
 * @param nx Width
 * @param ny Height
 * @param pitchf1 Pitch
 */
__global__ void subKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
);

/*!
 * @brief Sets the value of all positions in the field to the provided value
 * @param field_g Image/Field/Array
 * @param nx Width
 * @param ny Height
 * @param pitchf1 Pitch
 * @param value Provided value
 */
__global__ void setKernel
(
		float *field_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float value
);
#endif


#endif




