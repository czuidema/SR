/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    superresolution.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * superresolution.cuh
 *
 *  Created on: May 16, 2012
 *      Author: steinbrf
 */
#ifndef SUPERRESOLUTION_CUH
#define SUPERRESOLUTION_CUH
#include <vector>
#include <list>

/*!
 * Class for Storing a 2D displacement field on the GPU
 */
class FlowGPU
{
public:
	//! Frees the GPU Memory
	void clear();
	//! Horizontal Component
	float *u_g;
	//! Vertical Component
	float *v_g;
	//! Width
	int   nx;
	//! Height
	int   ny;
};


void computeSuperresolutionUngerGPU
(
		float *xi1_g,
		float *xi2_g,
		float *temp1_g,
		float *temp2_g,
		float *temp3_g,
		float *temp4_g,
		float *uor_g,
		float *u_g,
		std::vector<float*> &q_g,
		std::vector<float*> &images_g,
		std::list<FlowGPU> &flowsGPU,
		int   &nx,
		int   &ny,
		int   &pitchf1,
		int   &nx_orig,
		int   &ny_orig,
		int   &pitchf1_orig,
		int   &oi,
		float &tau_p,
		float &tau_d,
		float &factor_tv,
		float &huber_epsilon,
		float &factor_rescale_x,
		float &factor_rescale_y,
		float &blur,
		float &overrelaxation,
		int   debug
);



#endif
