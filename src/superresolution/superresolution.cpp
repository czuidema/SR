/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    superresolution.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * superresolution.cpp
 *
 *  Created on: Feb 29, 2012
 *      Author: steinbrf
 */

#include "superresolution.hpp"
#include "superresolution.cuh"
#include <stdio.h>
#include <iostream>

#include <filesystem/filesystem.h>

//#include <linearoperations.h>
#include <linearoperations/linearoperations.h>

#include <auxiliary/cuda_basic.cuh>

#include <auxiliary/debug.hpp>


void dualTVHuber
(
		const float *uor,
		float       *xi1,
		float       *xi2,
		int         nx,
		int         ny,
		float       factor_update,
		float       factor_clipping,
		float       huber_denom,
		float       tau_d
)
{
	for(int x=0;x<nx;x++){
		int x1 = x+1; if(x1 >= nx) x1 = nx-1;
		for(int y=0;y<ny;y++){
			 int y1 = y+1; if(y1 >= ny) y1 = ny-1;
			const int p = y*nx+x;
			float dx = (xi1[p] + tau_d * factor_update * (uor[y*nx+x1] - uor[p])) /huber_denom;
			float dy = (xi2[p] + tau_d * factor_update * (uor[y1*nx+x] - uor[p])) /huber_denom;
			float denom = sqrtf(dx*dx + dy*dy)/factor_clipping;
			if(denom < 1.0f) denom = 1.0f;
			xi1[p] = dx / denom;
			xi2[p] = dy / denom;
		}
	}
}

void dualL1Difference
(
		const float *primal,
		const float *constant,
		float       *dual,
		int         nx,
		int         ny,
		float       factor_update,
		float       factor_clipping,
		float       huber_denom,
		float       tau_d
)
{
	for(int p=0;p<nx*ny;p++){
		dual[p] = (dual[p] + tau_d*factor_update* (primal[p] - constant[p]))/huber_denom;
		if(dual[p] < -factor_clipping) dual[p] = -factor_clipping;
		if(dual[p] > factor_clipping)  dual[p] = factor_clipping;
	}
}

void primal1N
(
		const float *xi1,
		const float *xi2,
		const float *degraded,
		float       *u,
		float       *uor,
		int         nx,
		int         ny,
		float       factor_tv_update,
		float       factor_degrade_update,
		float       tau_p,
		float       overrelaxation
)
{
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			const int p = y*nx+x;
			float u_old = u[p];
			float u_new = u[p] + tau_p *
					(factor_tv_update *(xi1[p] - (x==0 ? 0.0f : xi1[p-1]) + xi2[p] - (y==0 ? 0.0f : xi2[p-nx]))
						 - factor_degrade_update * degraded[p]);
			u[p] = u_new;
			uor[p] = overrelaxation*u_new + (1.0f-overrelaxation) * u_old;
		}
	}
}


Flow::Flow()
: u1(NULL), u2(NULL), valid(false), saved(false), filename(""), src(0), dst(0),
smoothness(LAMBDA_INIT),
oi(OUTER_ITERATIONS_INIT), ii(INNER_ITERATIONS_INIT),
overrelaxation(OVERRELAXATION_SOR_INIT),
rescale_factor(0.5f),
levels(START_LEVEL+1)
{

}

Flow::Flow(int p_nx, int p_ny)
: u1(0), u2(0), valid(false), saved(false), filename(""), src(0), dst(0)
{
	set(p_nx,p_ny);
}

Flow::~Flow()
{

}

void Flow::clear()
{
	if(u1) delete [] u1;
	if(u2) delete [] u2;
}

void Flow::set(int p_nx, int p_ny)
{
	if(u1== NULL || u2 == NULL || nx != p_nx || ny != p_ny){
		nx = p_nx;
		ny = p_ny;
		if(u1) delete [] u1;
		if(u2) delete [] u2;
		u1 = new float[nx*ny];
		u2 = new float[nx*ny];
	}
}

void Flow::load_from_file()
{
	load_flow_file(filename,u1,u2,nx,ny);
}

void Flow::save_to_file()
{
	save_flow_file(filename,u1,u2,nx,ny);
}

SuperResolution::SuperResolution(int nx_orig, int ny_orig, int nx, int ny)
:
  _blur(0.5),
  _factor_tv(100.0),
  _tau_p(0.1),
  _tau_d(0.1),
  _iterations(20),
	_debug(0),
	_factor_rescale_x(0.0),
  _factor_rescale_y(0.0),
	_nx_orig(nx_orig),
	_ny_orig(ny_orig),
	_nx(nx),
	_ny(ny)
{
	_factor_rescale_x = (float)_nx / (float)_nx_orig;
	_factor_rescale_y = (float)_ny / (float)_ny_orig;
}



int SuperResolution::nx(){return _nx;}
int SuperResolution::ny(){return _ny;}
int SuperResolution::nx_orig(){return _nx_orig;}
int SuperResolution::ny_orig(){return _ny_orig;}


SuperResolution::~SuperResolution()
{
}

SuperResolution1N::SuperResolution1N(int nx_orig, int ny_orig, int nx, int ny)
: SuperResolution(nx_orig,ny_orig,nx,ny) ,_result(NULL)
{

}

SuperResolution1N::~SuperResolution1N()
{
	if(_result) delete _result;
}


SuperResolutionUnger::SuperResolutionUnger(int nx_orig, int ny_orig, int nx, int ny)
: SuperResolution1N(nx_orig,ny_orig,nx,ny),
	_huber_epsilon(0.0f), _overrelaxation(2.0f),
	_xi1(NULL), _xi2(NULL), _u_overrelaxed(NULL),
	_help1(NULL), _help2(NULL), _help3(NULL), _help4(NULL)
{
}

SuperResolutionUnger::~SuperResolutionUnger()
{}

SuperResolutionUngerCPU::SuperResolutionUngerCPU(int nx_orig, int ny_orig, int nx, int ny)
: SuperResolutionUnger(nx_orig,ny_orig,nx,ny)
{
	_xi1 = new float[_nx*_ny];
	_xi2 = new float[_nx*_ny];
	_help1 = new float[_nx*_ny];
	_help2 = new float[_nx*_ny];
	_help3 = new float[_nx*_ny];
	_help4 = new float[_nx*_ny];
	_u_overrelaxed = new float[_nx*_ny];
	_result = new cv::Mat(_ny,_nx,CV_32FC1);
}

SuperResolutionUngerCPU::~SuperResolutionUngerCPU()
{
	if(_xi1) delete [] _xi1;
	if(_xi2) delete [] _xi2;
	for(unsigned int i=0;i<_q.size();i++) if(_q[i]) delete [] _q[i];
	if(_help1) delete [] _help1;
	if(_help2) delete [] _help2;
	if(_help3) delete [] _help3;
	if(_help4) delete [] _help4;

	if(_u_overrelaxed) delete [] _u_overrelaxed;
}

void SuperResolutionUngerCPU::compute()
{
	fprintf(stderr,"\nComputing 1N Superresolution from %i Images on CPU",(int)_images_original.size());

	for(unsigned int i=_q.size();i<_images_original.size();i++){
		_q.push_back(new float[_nx_orig*_ny_orig]);
	}

	float *u = (float*)_result->data;
//	float *uor = u;

	for(int x=0;x<_nx*_ny;x++){
		_xi1[x] = _xi2[x] = 0.0f;
		u[x] = _u_overrelaxed[x] = 64.0f;
	}
	for(unsigned int k=0;k<_q.size();k++){
		for(int x=0;x<_nx_orig*_ny_orig;x++){
			_q[k][x] = 0.0f;
		}
	}

	float factorquad = _factor_rescale_x*_factor_rescale_y*_factor_rescale_x*_factor_rescale_y;
	float factor_degrade_update = pow(factorquad,CLIPPING_TRADEOFF_DEGRADE_1N);
	float factor_degrade_clipping = factorquad/factor_degrade_update;
	float huber_denom_degrade = 1.0f + _huber_epsilon*_tau_d/factor_degrade_clipping;


	float factor_tv_update = pow(_factor_tv,CLIPPING_TRADEOFF_TV);
	float factor_tv_clipping = _factor_tv/factor_tv_update;
	float huber_denom_tv = 1.0f + _huber_epsilon*_tau_d/_factor_tv;

	for(int i=0;i<_iterations;i++){
		fprintf(stderr," %i",i);

		//DUAL TV
		dualTVHuber(_u_overrelaxed,_xi1,_xi2,_nx,_ny,factor_tv_update,factor_tv_clipping,huber_denom_tv,_tau_d);

		//DUAl DATA
		unsigned int k=0;
		std::vector<cv::Mat*>::iterator image = _images_original.begin();
		std::list<Flow>::iterator flow = _flows.begin();
		while(image != _images_original.end() && flow != _flows.end() && k < _q.size()){
			if(_factor_rescale_x > 1.0f || _factor_rescale_y > 1.0f){
				resampleAreaParallelizableSeparateAdjoined(_q[k],_help1,_nx_orig,_ny_orig,_nx,_ny,_help4);
			}
			else{
				memcpy(_help1,_q[k],_nx_orig*_ny_orig*sizeof(float));
			}
			if(_blur > 0.0f){
				gaussBlurSeparateMirror(_help1,_help2,_nx,_ny,_blur,_blur,(int)(3.0f*_blur),_help4,0);
			}
			else{
				float *temp = _help1; _help1 = _help2; _help2 = temp;
			}
			forewardRegistrationBilinear(_help2,_help1,flow->u1,flow->u2,_nx,_ny);
			for(int x=0;x<_nx*_ny;x++) _help3[x] += _help1[x];
			k++;
			flow++;
			image++;
		}

		//PROX
		for(int x=0;x<_nx*_ny;x++) _help3[x] = 0.0f;
		k=0;
		image = _images_original.begin();
		flow = _flows.begin();
		while(image != _images_original.end() && flow != _flows.end() && k < _q.size()){
			resampleAreaParallelizableSeparateAdjoined(_q[k],_help1,_nx_orig,_ny_orig,_nx,_ny,_help4);
			if(_blur > 0.0f){
				gaussBlurSeparateMirror(_help1,_help2,_nx,_ny,_blur,_blur,(int)(3.0f*_blur),_help4,0);
			}
			else{
				float *temp = _help1; _help1 = _help2; _help2 = temp;
			}
			if(_factor_rescale_x > 1.0f || _factor_rescale_y > 1.0f){
				forewardRegistrationBilinear(_help2,_help1,flow->u1,flow->u2,_nx,_ny);
			}
			else{
				float *temp = _help1; _help1 = _help2; _help2 = temp;
			}
			for(int x=0;x<_nx*_ny;x++) _help3[x] += _help1[x];
			k++;
			flow++;
			image++;
		}

		primal1N(_xi1,_xi2,_help3,u,_u_overrelaxed,_nx,_ny,factor_tv_update,factor_degrade_update,_tau_p,_overrelaxation);

	}
}


SuperResolutionUngerGPU::SuperResolutionUngerGPU(int nx_orig, int ny_orig, int nx, int ny)
: SuperResolutionUnger(nx_orig,ny_orig,nx,ny),
  _pitchf1_orig(0), _pitchf1(0), _result_g(NULL)
{
	cuda_malloc2D((void**)&_xi1,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_xi2,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_help1,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_help2,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_help3,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_help4,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_u_overrelaxed,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_result_g,_nx,_ny,1,sizeof(float),&_pitchf1);
	printf("\nTHE PITCH SIZE IS: %d ", _pitchf1);
	printf("\nTHE SIZE OF THE RESULT IS: %d\n", _pitchf1*nx*ny);
	_result = new cv::Mat(_ny,_nx,CV_32FC1);
}

SuperResolutionUngerGPU::~SuperResolutionUngerGPU()
{
	if(_xi1) cuda_free(_xi1);
	if(_xi2) cuda_free(_xi2);
	if(_u_overrelaxed) cuda_free(_u_overrelaxed);
	if(_result_g) cuda_free(_result_g);
	for(unsigned int i=0;i<_q.size();i++) if(_q[i]) cuda_free(_q[i]);
	if(_help1) cuda_free(_help1);
	if(_help2) cuda_free(_help2);
	if(_help3) cuda_free(_help3);
	if(_help4) cuda_free(_help4);
}


void FlowGPU::clear()
{
	if(u_g) cuda_free(u_g);
	if(v_g) cuda_free(v_g);
}

void SuperResolutionUngerGPU::compute()
{

  float *u = (float*)_result->data;
  for(int x=0;x<_nx*_ny;x++){
    u[x] = (float)(x%256);
  }

  float *memhelp;
  compute_pitch_alloc((void**)&memhelp,_nx_orig,_ny_orig,1,sizeof(float),&_pitchf1_orig);
  int memoryFlow = _flows.size()*_pitchf1*_ny*sizeof(float)*2;
  int memoryDegrade = _flows.size()*_pitchf1_orig*_ny_orig*sizeof(float);
  int memoryImages = _flows.size()*_pitchf1_orig*_ny_orig*sizeof(float);
  fprintf(stderr,"\nNeed %i Bytes (%i MB) of Memory on GPU for Flows,"
      "%i Bytes (%i MB) for Low-Res Images, %i Bytes (%i MB) for Warp Dual Variables",
      memoryFlow,memoryFlow/1048576+1,
      memoryDegrade,memoryDegrade/1048576+1,
      memoryImages,memoryImages/1048576+1);

  std::vector<float*>images_g;
  for(std::vector<cv::Mat*>::iterator i=_images_original.begin();i!=_images_original.end();i++){
    float *temp;
    cuda_malloc2D((void**)(&temp),_nx_orig,_ny_orig,1,sizeof(float),&_pitchf1_orig);
    cuda_copy_h2d_2D((float*)((*i)->data),temp,_nx_orig,_ny_orig,1,sizeof(float),_pitchf1_orig);
    images_g.push_back(temp);
  }

  for(std::list<Flow>::iterator i=_flows.begin();i!=_flows.end();i++){
    FlowGPU temp;
    cuda_malloc2D((void**)(&(temp.u_g)),_nx,_ny,1,sizeof(float),&_pitchf1);
    cuda_malloc2D((void**)(&(temp.v_g)),_nx,_ny,1,sizeof(float),&_pitchf1);
    cuda_copy_h2d_2D(i->u1,temp.u_g,_nx,_ny,1,sizeof(float),_pitchf1);
    cuda_copy_h2d_2D(i->u2,temp.v_g,_nx,_ny,1,sizeof(float),_pitchf1);
    temp.nx = _nx; temp.ny = _ny;
    _flowsGPU.push_back(temp);
  }

  for(unsigned int i=_q.size();i<images_g.size();i++){
    _q.push_back(NULL);
    cuda_malloc2D((void**)&(_q.back()),_nx_orig,_ny_orig,1,sizeof(float),&_pitchf1_orig);
  }


  computeSuperresolutionUngerGPU(_xi1,_xi2,_help1,_help2,_help3,_help4,_u_overrelaxed,_result_g,_q,images_g,_flowsGPU,
      _nx,_ny,_pitchf1,_nx_orig,_ny_orig,_pitchf1_orig,_iterations,_tau_p,_tau_d,_factor_tv,_huber_epsilon,_factor_rescale_x,_factor_rescale_y,_blur,_overrelaxation,_debug);

  for(std::vector<float*>::iterator i=images_g.begin();i!=images_g.end();i++){
    cuda_free(*i);
  }

  for(std::list<FlowGPU>::iterator i=_flowsGPU.begin();i!=_flowsGPU.end();i++){
    i->clear();
  }

  //Copy back result
  cuda_copy_d2h_2D(_result_g,u,_nx,_ny,1,sizeof(float),_pitchf1);


}


void computeFlowsOnVector
(
		FlowLib *flowlib,
		std::vector<std::list<Flow> > &flows,
		std::vector<float*> &flowInputImages,
		int nx,
		int ny,
		int time_range,
		int iteration
)
{
	fprintf(stderr,"\nComputing all Flows");
	for(int k=0;(unsigned int)k<flowInputImages.size();k++){
		int j = k-time_range; if(j<0) j = 0; // b
		for(std::list<Flow>::iterator it = flows[k].begin();it!=flows[k].end();it++){
			it->set(nx,ny);
			flowlib->setInput(flowInputImages[k]);
			flowlib->setInput(flowInputImages[j]);
			flowlib->computeFlow();
			flowlib->getOutput(it->u1,it->u2);
			if(iteration >= 0){
				it->save_to_file();
			}
			j++;
		}
	}
}

