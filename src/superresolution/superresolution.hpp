/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    superresolution.hpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * superresolution.h
 *
 *  Created on: Feb 29, 2012
 *      Author: steinbrf
 */

#ifndef SUPERRESOLUTION_H_
#define SUPERRESOLUTION_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <list>

#include <flowlib/flowio.h>
//#include <resampling_old.h>
#include <superresolution/superresolution.cuh>

#include <superresolution/superresolution_definitions.h>
#include <flowlib/flowlib.hpp>

/*!
 * Class for Storing a 2D displacement field and the parameters
 * used for creating it
 */
class Flow
{
public:
	//! Constructor
	Flow();
	//! Constructor
	Flow(int p_nx, int p_ny);
	//! Destructor
	~Flow();
	//! Frees the field memories
	void clear();
	//! Sets the dimensions
	void set(int p_nx, int p_ny);
	//! Loads the flow from a file speciefied by filename
	void load_from_file();
	//! Saves the flow to a file specified by filename
	void save_to_file();
	//! Horizontal component
	float *u1;
	//! Vertical component
	float *u2;
	//! Width
	int   nx;
	//! Height
	int   ny;
	//! Saves whether the flow has been computed or loaded
	bool valid;
	//! Saves whether the flow has been saved to a file
	bool saved;
	//! Filename
	std::string filename;
	//! Pointer to first image
	cv::Mat *src;
	//! Pointer to second image
	cv::Mat *dst;
	//! Smoothness weight
	float smoothness;
	//! Outer Iterations
	int   oi;
	//! Inner Iterations
	int   ii;
	//! Overrelaxation
	float overrelaxation;
	//! Rescale factor for coarse-to-fine implementation
	float rescale_factor;
	//! Number of Levels
	int   levels;
};


/*!
 * Base Class for Superresolution
 */
class SuperResolution
{
public:
	//! Constructor
	SuperResolution(int nx_orig, int ny_orig, int nx, int ny);
	//! Destructor
	virtual ~SuperResolution();
	//! Compute Function
	virtual void compute() = 0;

	//! Getter function for the original width
	int   nx_orig();

	//! Getter function for the original height
	int   ny_orig();

	//! Getter function for the high resolution width
	int   nx();

	//! Getter function for the high resolution height
	int   ny();

	//! The amount of Gaussian Blur present in the degrading process
	float _blur;

	//! The weight of Total Variation Penalization
	float _factor_tv;

	//! Primal Update Step Size
	float _tau_p;

	//! Dual Update Step Size
	float _tau_d;

	//! Number of Iterations
	int   _iterations;

	//! Debug Flag, if activated the class produces Debug output.
	int _debug;

	//! Vector of Pointers to the original Low-Resolution Images
	std::vector<cv::Mat*> _images_original;

protected:
	//! High-Resolution Width divided by Low-Resolution Width
	float _factor_rescale_x;

	//! High-Resolution Height divided by Low-Resolution Height
	float _factor_rescale_y;

	//! Original Low-Resolution Width
	int   _nx_orig;

	//! Original Low-Resolution Height
	int   _ny_orig;

	//! New High-Resolution Width
	int   _nx;

	//! New High-Resolution Height
	int   _ny;
};

/*!
 * Class for the computation of separately computing 1
 * high resolution output image from N low resolution
 * input images
 */
class SuperResolution1N : public SuperResolution
{
public:
	//! Constructor
	SuperResolution1N(int nx_orig, int ny_orig, int nx, int ny);
	//! Destructor
	virtual ~SuperResolution1N();

	//! Resulting High-Resolution Image
	cv::Mat *_result;

	//! Vector of Flows from the image to be computed in high Resolution to the other images in the given time range
	std::list<Flow> _flows;
};

/*!
 * Class for Superresolution computation by the method of Unger et al.
 * (http://gpu4vision.icg.tugraz.at/index.php?content=Cat_0&id=50&field=Paper#pub50)
 */
class SuperResolutionUnger : public SuperResolution1N
{
public:
	//! Constructor
	SuperResolutionUnger(int nx_orig, int ny_orig, int nx, int ny);
	//! Destructor
	virtual ~SuperResolutionUnger();

protected:
	//! Parameter for Huber norm regularization
	float _huber_epsilon;

	//! Overrelaxation parameter in the range of [1,2]
	float _overrelaxation;

	//! Dual Variable for TV regularization in X direction
	float *_xi1;

	//! Dual Variable for TV regularization in X direction
	float *_xi2;

	//! Field of overrelaxed primal variables
	float *_u_overrelaxed;

	//! Dual variables for L1 difference penalization
	std::vector<float*> _q;

	//! Helper array
	float *_help1;
	//! Helper array
	float *_help2;
	//! Helper array
	float *_help3;
	//! Helper array
	float *_help4;
};

/*!
 * CPU implementation of the Superresolution method of Unger et al.
 */
class SuperResolutionUngerCPU : public SuperResolutionUnger
{
public:
	//! Constructor
	SuperResolutionUngerCPU(int nx_orig, int ny_orig, int nx, int ny);
	//! Destructor
	virtual ~SuperResolutionUngerCPU();
	//! Compute Function
	void compute();

protected:
};

/*!
 * GPU implementation of the Superresolution method of Unger et al.
 */
class SuperResolutionUngerGPU : public SuperResolutionUnger
{
public:
	//! Constructor
	SuperResolutionUngerGPU(int nx_orig, int ny_orig, int nx, int ny);
	//! Destructor
	virtual ~SuperResolutionUngerGPU();
	//! Compute Function
	void compute();

protected:

	//! GPU pitch (padded width) of the original low-res images
	int _pitchf1_orig;

	//! GPU pitch (padded width) of the superresolution high-res fields
	int _pitchf1;

	//! GPU memory for the result image
	float *_result_g;

	//! GPU memory for the displacement fields
	std::list<FlowGPU> _flowsGPU;
};

/*!
 * @brief Computes the flows between all images in the vector and saves the to the flow vectors
 * @param flowlib
 * @param flows
 * @param flowInputImages
 * @param nx
 * @param ny
 * @param time_range
 * @param iteration
 */
void computeFlowsOnVector
(
		FlowLib *flowlib,
		std::vector<std::list<Flow> > &flows,
		std::vector<float*> &flowInputImages,
		int nx,
		int ny,
		int time_range,
		int iteration = -1
);


#endif /* SUPERRESOLUTION_H_ */
