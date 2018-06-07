/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    flowlib.hpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * flowlib.hpp
 *
 *  Created on: Mar 13, 2012
 *      Author: steinbrf
 */

#ifndef FLOWLIB_HPP_
#define FLOWLIB_HPP_

#define FLOW_DEBUG 0
#define START_LEVEL 5
#define END_LEVEL 0
#define OUTER_ITERATIONS_INIT 50
#define INNER_ITERATIONS_INIT 5
#define WARP_ITERATIONS_INIT 1
#define LAMBDA_INIT 10.0f
#define THETA_INIT 0.1f
#define TAU_P_INIT 0.1f
#define TAU_D_INIT 2.45f
#define PRESMOOTH_SIGMA_INIT 0.0f
#define SHOW_SCALE_INIT 3.0f
#define SCALE_FACTOR 0.5f
#define OVERRELAXATION_SOR_INIT 1.0f
#define OVERRELAXATION_PD_INIT 0.0f
#define PRIMALDUAL_METHOD 0 //0 = Decoupling, 1 = Relaxation
//#define SOR_DAT_EPSILON_INIT 0.001f
//#define SOR_REG_EPSILON_INIT 0.001f
#define SOR_DAT_EPSILON_INIT 0.1f
#define SOR_REG_EPSILON_INIT 0.1f

//#include <imagepyramidcpu.hpp>
//#include <imagepyramidgpu.hpp>
#include <imagepyramid/imagepyramid.hpp>

/*!
 * Base class for Optical Flow Computation
 */
class FlowLib
{
public:
	//! Constructor
	FlowLib(int par_nx, int par_ny);
	//! Destructor
	virtual ~FlowLib();
	//! Computation Method
	virtual float computeFlow() = 0;

	//! Computation of the color-coded flow image in RGB
	virtual void getOutputRGB
	(
			float *output,
			float pmin,
			float pmax,
			float ptmin,
			float factor
	) = 0;

	//! Computation of the color-coded flow image in BGR
	virtual void getOutputBGR
	(
			float *output,
			float pmin,
			float pmax,
			float ptmin,
			float factor
	) = 0;

	//! Sets another input image, flows are computed from 1 to 2, 3 to 4, and so on
	virtual void setInput(float *input);

	//! Retrieves the computed flow and copies it to par_u1 and par_u2
	virtual void getOutput(float *par_u1, float *par_u2);

	//! Returns a pointer to the internal horizontal flow component
	const float *horizontal();

	//! Returns a pointer to the internal vertical flow component
	const float *vertical();

	//! Setter for the smoothness weight
	void setLambda(float par_lambda);

	//! Setter for the overrelaxation paramter
	void setOverrelaxation(float par_overrelaxation);

	//! Setter for the outer iterations
	void setOuterIterations(int par_oi);

	//! Setter for the inner iterations
	void setInnerIterations(int par_ii);

	//! Setter for the start level
	void setStartLevel(int par_start_level);

	//! Setter for the end level
	void setEndLevel(int par_end_level);

protected:

	//! Computes the maximum Levels
	virtual int computeMaxWarpLevels() = 0;

	//! Pointer to the externally set first image
	float *_I1;

	//! Pointer to the externally set second image
	float *_I2;

	//! Horizontal component of the flow
	float *_u1;

	//! Vertical component of the flow
	float *_u2;

	// Members for Coarse-to-Fine Warping
	//! Warped second image
	float *_I2warp;

	//! Increment of the horizontal flow component
	float *_u1lvl;

	//! Increment of the vertical flow component
	float *_u2lvl;

	//! Width
	unsigned int _nx;

	//! Height
	unsigned int _ny;

	//! Smoothness weight
	float _lambda;

	//! Overrelaxation parameter in the range [1,2]
	float _overrelaxation;

	//! Outer iterations for lagged nonlinearity
	unsigned int _oi;

	//! Inner iterations for SOR
	unsigned int _ii;

	//! Scale of the flow for color-coding
	float _show_scale;

	//! Threshold of the flow for color-coding
	float _show_threshold;

	//! Start level
	int _start_level;

	//! End level
	int _end_level;

	//! Verbose parameter, if set the class prints what it does
	int _verbose;
	//! Debug parameter, if se the class produces debug output
	int _debug;
	//! Output buffer for debug output
	char *_debugbuffer;
};

/*!
 * Class for CPU implementations of the optical flow
 */
class FlowLibCpu : virtual public FlowLib
{
public:
	//! Constructor
	FlowLibCpu(int par_nx, int par_ny);
	//! Destructor
	virtual ~FlowLibCpu();

	//! Computation of the color-coded flow image in RGB
	void getOutputRGB
	(
			float *output,
			float pmin,
			float pmax,
			float ptmin,
			float factor
	);

	//! Computation of the color-coded flow image in BGR
	void getOutputBGR
	(
			float *output,
			float pmin,
			float pmax,
			float ptmin,
			float factor
	);

	//! Sets another input image, flows are computed from 1 to 2, 3 to 4, and so on
	void setInput(float *input);

protected:

	//! Derivative of the Images in X direction
	float *_Ix;
	//! Derivative of the Images in Y direction
	float *_Iy;
	//! Derivative of the Images in temporal direction
	float *_It;

	//! Image pyramid for the first image
	ImagePyramidCpu *_I1pyramid;
	//! Image pyramid for the second image
	ImagePyramidCpu *_I2pyramid;
};

/*!
 * Class for GPU implementations of the optical flow
 */
class FlowLibGpu : virtual public FlowLib
{
public:
	//! Constructor
	FlowLibGpu(int par_nx, int par_ny);
	//! Destructor
	virtual ~FlowLibGpu();

	//! Computation of the color-coded flow image in RGB
	void getOutputRGB
	(
			float *output,
			float pmin,
			float pmax,
			float ptmin,
			float factor
	);

	//! Computation of the color-coded flow image in BGR
	void getOutputBGR
	(
			float *output,
			float pmin,
			float pmax,
			float ptmin,
			float factor
	);

	//! Sets another input image, flows are computed from 1 to 2, 3 to 4, and so on
	void setInput(float *input);

	//! Retrieves the computed flow and copies it to par_u1 and par_u2
	void getOutput(float *par_u1, float *par_u2);

protected:

	//! Image pyramid for the first image
	ImagePyramidGpu *_I1pyramid;
	//! Image pyramid for the second image
	ImagePyramidGpu *_I2pyramid;

	//! GPU memory for the horizontal component
	float *_u1_g;
	//! GPU memory for the vertical component
	float *_u2_g;

	//! GPU memory for the color-coded flow
	float *_colorOutput_g;

	//! Pitch for scalar-valued fields
	int _pitchf1;
	//! Pitch for RGB-valued fields
	int _pitchf3;

	//! Saves whether the first image has been set
	bool firstImageSet;
};

/*!
 * Class for SOR implementations of the optical flow
 */
class FlowLibSOR : virtual public FlowLib
{
public:
	//! Constructor
	FlowLibSOR(int par_nx, int par_ny);
	//! Destructor
	virtual ~FlowLibSOR();

protected:

	//! Computes the maximum Number of Levels with default rescale factor
	int computeMaxWarpLevels();
	//! Computes the maximum Number of Levels for a specified rescale factor
	int computeMaxWarpLevels(float rescale_factor);

	//! Epsilon for differentiability for L1 dataterm penalization
	float _dat_epsilon;
	//! Epsilon for differentiability for TV regularization
	float _reg_epsilon;

	//! Memory for lagged nonlinearity of the dataterm
	float *_penDat;
	//! Memory for lagged nonlinearity of the regularity term
	float *_penReg;

	//! Memory for the right-hand side of the intermediate linear equations of the horizontal component
	float *_b1;
	//! Memory for the right-hand side of the intermediate linear equations of the vertical component
	float *_b2;
};



//##################################################################
/*!
 * Class for CPU SOR implementations of the optical flow
 */
class FlowLibCpuSOR : public FlowLibCpu, public FlowLibSOR
{
public:
	//! Constructor
	FlowLibCpuSOR(int par_nx, int par_ny);
	//! Destructor
	virtual ~FlowLibCpuSOR();
	//! Computation Method
	float computeFlow();
protected:
};

/*!
 * Class for GPU SOR implementations of the optical flow
 */
class FlowLibGpuSOR : public FlowLibGpu, public FlowLibSOR
{
public:
	//! Constructor
	FlowLibGpuSOR(int par_nx, int par_ny);
	//! Destructor
	virtual ~FlowLibGpuSOR();
	//! Computation Method
	float computeFlow();
protected:
};




#endif /* FLOWLIB_HPP_ */
