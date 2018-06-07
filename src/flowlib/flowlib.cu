/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    flowlib.cu
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * flowlib.cu
 *
 *  Created on: Mar 20, 2012
 *      Author: steinbrf
 */

#include "flowlib.hpp"
#include <auxiliary/cuda_basic.cuh>

#define SF_BW 16
#define SF_BH 16

__global__ void sorflow_hv_to_rgb_kernel
(
		float *u_g,
		float *v_g,
		float3 *rgb_g,
		float  pmin,
		float  pmax,
		float  ptmin,
		int    nx,
		int    ny,
		float  factor,
		int    pitchf1,
		int    pitchf3
);
__global__ void sorflow_hv_to_bgr_kernel
(
		float *u_g,
		float *v_g,
		float3 *rgb_g,
		float  pmin,
		float  pmax,
		float  ptmin,
		int    nx,
		int    ny,
		float  factor,
		int    pitchf1,
		int    pitchf3
);

FlowLibGpu::FlowLibGpu(int par_nx, int par_ny):
FlowLib(par_nx,par_ny),
_u1_g(NULL), _u2_g(NULL), _colorOutput_g(NULL),_pitchf1(0),_pitchf3(0),
firstImageSet(false)
{
	fprintf(stderr,"\nFlowLibGpu: Allocating Image Pyramids...");
	_I1pyramid = new ImagePyramidGpu(_nx,_ny,1,SCALE_FACTOR);
	_I2pyramid = new ImagePyramidGpu(_nx,_ny,1,SCALE_FACTOR);
	fprintf(stderr,"\nFlowLibGpu: Allocating Fields...");
	cuda_malloc2D((void**)&_u1_g,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_u2_g,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_I2warp,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_u1lvl,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_u2lvl,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_colorOutput_g,_nx,_ny,1,sizeof(float3),&_pitchf3);
	fprintf(stderr,"\nFlowLibGpu: Fields Allocated.");
}

FlowLibGpu::~FlowLibGpu()
{
	if(_u1_g) cutilSafeCall(cudaFree(_u1_g));
	if(_u2_g) cutilSafeCall(cudaFree(_u2_g));
	if(_I2warp) cutilSafeCall(cudaFree(_I2warp));
	if(_u1lvl) cutilSafeCall(cudaFree(_u1lvl));
	if(_u2lvl) cutilSafeCall(cudaFree(_u2lvl));
	if(_colorOutput_g) cutilSafeCall(cudaFree(_colorOutput_g));
}

void FlowLibGpu::getOutputRGB
(
		float *output,
		float pmin,
		float pmax,
		float ptmin,
		float factor
)
{
	int ngx = (_nx%SF_BW) ? ((_nx/SF_BW)+1) : (_nx/SF_BW);
	int ngy = (_ny%SF_BH) ? ((_ny/SF_BH)+1) : (_ny/SF_BH);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);

	sorflow_hv_to_rgb_kernel<<<dimGrid,dimBlock>>>
			(_u1_g,_u2_g,(float3*)_colorOutput_g,pmin,pmax,ptmin,
					_nx,_ny,factor,_pitchf1,_pitchf3);
	catchkernel;
	cuda_copy_d2h_2D(_colorOutput_g,output,_nx,_ny,3,sizeof(float),_pitchf3);
}
void FlowLibGpu::getOutputBGR
(
		float *output,
		float pmin,
		float pmax,
		float ptmin,
		float factor
)
{
	int ngx = (_nx%SF_BW) ? ((_nx/SF_BW)+1) : (_nx/SF_BW);
	int ngy = (_ny%SF_BH) ? ((_ny/SF_BH)+1) : (_ny/SF_BH);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);

	sorflow_hv_to_bgr_kernel<<<dimGrid,dimBlock>>>
			(_u1_g,_u2_g,(float3*)_colorOutput_g,pmin,pmax,ptmin,
					_nx,_ny,factor,_pitchf1,_pitchf3);
	catchkernel;
	cuda_copy_d2h_2D(_colorOutput_g,output,_nx,_ny,3,sizeof(float),_pitchf3);
}

void FlowLibGpu::setInput(float *input)
{
	if(!firstImageSet){
		cuda_copy_h2d_2D(input,_I2pyramid->level[0],_nx,_ny,1,sizeof(float),_pitchf1);
		_I2pyramid->fill();
		firstImageSet = true;
	}

	ImagePyramidGpu *temp = _I1pyramid;
	_I1pyramid = _I2pyramid;
	_I2pyramid = temp;
	cuda_copy_h2d_2D(input,_I2pyramid->level[0],_nx,_ny,1,sizeof(float),_pitchf1);
	_I2pyramid->fill();
}

void FlowLibGpu::getOutput(float *par_u1, float *par_u2)
{
	cuda_copy_d2h_2D(_u1_g,_u1,_nx,_ny,1,sizeof(float),_pitchf1);
	cuda_copy_d2h_2D(_u2_g,_u2,_nx,_ny,1,sizeof(float),_pitchf1);
	// TODO: Hier lieber memcpy verwenden
	for(int p=0;p<_nx*_ny;p++){
		par_u1[p] = _u1[p];
		par_u2[p] = _u2[p];
	}
}































//TODO : ZU einem Kernel zusammenfuegen, der die Posen von R,G,B parametrisiert
__global__ void sorflow_hv_to_rgb_kernel
(
		float *u_g,
		float *v_g,
		float3 *rgb_g,
		float  pmin,
		float  pmax,
		float  ptmin,
		int    nx,
		int    ny,
		float  factor,
		int    pitchf1,
		int    pitchf3
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float temp;
	float u;
	float v;
	float3 rgb;


	float Pi;
	float amp;
	float phi;
	float alpha, beta;

	if (x < nx && y < ny){
		u = u_g[y*pitchf1+x];
		v = v_g[y*pitchf1+x];
	}


	/* set pi */
	Pi = 2.0f * acos(0.0f);

	/* determine amplitude and phase (cut amp at 1) */
	amp = (sqrtf(u*u + v*v) - pmin)/(pmax - pmin);
	if (amp > 1.0f) amp = 1.0f;
	if (u == 0.0f)
		if (v >= 0.0f) phi = 0.5f * Pi;
		else phi = 1.5f * Pi;
	else if (u > 0.0f)
		if (v >= 0.0f) phi = atanf(v/u);
		else phi = 2.0f * Pi + atanf(v/u);
	else phi = Pi + atanf (v/u);

	phi = phi / 2.0f;

	// interpolation between red (0) and blue (0.25 * Pi)
	if ((phi >= 0.0f) && (phi < 0.125f * Pi)) {
		beta  = phi / (0.125f * Pi);
		alpha = 1.0f - beta;
		rgb.x = amp * (alpha+ beta);
		rgb.y = 0.0f;
		rgb.z = amp * beta;
	}
	else if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
		beta  = (phi-0.125 * Pi) / (0.125 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (alpha + beta *  0.25f);
		rgb.y = amp * (beta *  0.25f);
		rgb.z = amp * (alpha + beta);
	}
	// interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
	else if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
		beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (alpha *  0.25f);
		rgb.y = amp * (alpha *  0.25f + beta);
		rgb.z = amp * (alpha+ beta);
	}
	else if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
		beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
		alpha = 1.0 - beta;
		rgb.x = 0.0f;
		rgb.y = amp * (alpha+ beta);
		rgb.z = amp * (alpha);
	}
	// interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
	else if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
		beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (beta);
		rgb.y = amp * (alpha+ beta);
		rgb.z = 0.0f;
	}
	// interpolation between yellow (0.75 * Pi) and red (Pi)
	else if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
		beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (alpha+ beta);
		rgb.y = amp * (alpha);
		rgb.z = 0.0f;
	}
	else {
		rgb.x = rgb.y = rgb.z = 0.5f;
	}

	temp = atan2f(-u,-v)*0.954929659f; // 6/(2*PI)
	if (temp < 0) temp += 6.0f;

	rgb.x = (rgb.x>=1.0f)*1.0f+((rgb.x<1.0f)&&(rgb.x>0.0f))*rgb.x;
	rgb.y = (rgb.y>=1.0f)*1.0f+((rgb.y<1.0f)&&(rgb.y>0.0f))*rgb.y;
	rgb.z = (rgb.z>=1.0f)*1.0f+((rgb.z<1.0f)&&(rgb.z>0.0f))*rgb.z;

	rgb.x *= factor; rgb.y *= factor; rgb.z *= factor;

	if (x < nx && y < ny)
		rgb_g[y*pitchf3+x] = rgb;
}

__global__ void sorflow_hv_to_bgr_kernel
(
		float *u_g,
		float *v_g,
		float3 *rgb_g,
		float  pmin,
		float  pmax,
		float  ptmin,
		int    nx,
		int    ny,
		float  factor,
		int    pitchf1,
		int    pitchf3
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float temp;
	float u;
	float v;
	float3 rgb;


	float Pi;
	float amp;
	float phi;
	float alpha, beta;

	if (x < nx && y < ny){
		u = u_g[y*pitchf1+x];
		v = v_g[y*pitchf1+x];
	}


	/* set pi */
	Pi = 2.0f * acos(0.0f);

	/* determine amplitude and phase (cut amp at 1) */
	amp = (sqrtf(u*u + v*v) - pmin)/(pmax - pmin);
	if (amp > 1.0f) amp = 1.0f;
	if (u == 0.0f)
		if (v >= 0.0f) phi = 0.5f * Pi;
		else phi = 1.5f * Pi;
	else if (u > 0.0f)
		if (v >= 0.0f) phi = atanf(v/u);
		else phi = 2.0f * Pi + atanf(v/u);
	else phi = Pi + atanf (v/u);

	phi = phi / 2.0f;

	// interpolation between red (0) and blue (0.25 * Pi)
	if ((phi >= 0.0f) && (phi < 0.125f * Pi)) {
		beta  = phi / (0.125f * Pi);
		alpha = 1.0f - beta;
		rgb.x = amp * (alpha+ beta);
		rgb.y = 0.0f;
		rgb.z = amp * beta;
	}
	else if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
		beta  = (phi-0.125 * Pi) / (0.125 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (alpha + beta *  0.25f);
		rgb.y = amp * (beta *  0.25f);
		rgb.z = amp * (alpha + beta);
	}
	// interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
	else if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
		beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (alpha *  0.25f);
		rgb.y = amp * (alpha *  0.25f + beta);
		rgb.z = amp * (alpha+ beta);
	}
	else if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
		beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
		alpha = 1.0 - beta;
		rgb.x = 0.0f;
		rgb.y = amp * (alpha+ beta);
		rgb.z = amp * (alpha);
	}
	// interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
	else if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
		beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (beta);
		rgb.y = amp * (alpha+ beta);
		rgb.z = 0.0f;
	}
	// interpolation between yellow (0.75 * Pi) and red (Pi)
	else if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
		beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
		alpha = 1.0 - beta;
		rgb.x = amp * (alpha+ beta);
		rgb.y = amp * (alpha);
		rgb.z = 0.0f;
	}
	else {
		rgb.x = rgb.y = rgb.z = 0.5f;
	}

	temp = atan2f(-u,-v)*0.954929659f; // 6/(2*PI)
	if (temp < 0) temp += 6.0f;

	rgb.x = (rgb.x>=1.0f)*1.0f+((rgb.x<1.0f)&&(rgb.x>0.0f))*rgb.x;
	rgb.y = (rgb.y>=1.0f)*1.0f+((rgb.y<1.0f)&&(rgb.y>0.0f))*rgb.y;
	rgb.z = (rgb.z>=1.0f)*1.0f+((rgb.z<1.0f)&&(rgb.z>0.0f))*rgb.z;

	temp = rgb.x*factor;

	rgb.x = rgb.z*factor; rgb.y *= factor; rgb.z = temp;


	if (x < nx && y < ny)
		rgb_g[y*pitchf3+x] = rgb;
}









