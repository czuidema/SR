/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    flowlib.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * flowlib.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: steinbrf
 */

#include "flowlib.hpp"
#include <math.h>
#include <stdio.h>
#include <vector>
#include <list>



/*########################## FlowLib ################################*/


FlowLib::FlowLib(int par_nx, int par_ny):
_I1(0),
_I2(0),
_u1(0),
_u2(0),
_I2warp(0),
_u1lvl(0),
_u2lvl(0),
_nx(par_nx),
_ny(par_ny),
_lambda(LAMBDA_INIT),
_overrelaxation(OVERRELAXATION_PD_INIT),
_oi(OUTER_ITERATIONS_INIT),
_ii(INNER_ITERATIONS_INIT),
_show_scale(1.0f),
_show_threshold(0.0f),
_start_level(START_LEVEL),
_end_level(END_LEVEL),
_verbose(0),
_debug(FLOW_DEBUG)
{
	_u1 = new float[_nx*_ny];
	_u2 = new float[_nx*_ny];

	_I2warp = new float[_nx*_ny];
	_u1lvl = new float[_nx*_ny];
	_u2lvl = new float[_nx*_ny];


  if(_debug) _debugbuffer = new char[100];
}

FlowLib::~FlowLib()
{
	if(_u1) delete [] _u1;
	if(_u2) delete [] _u2;
}

void FlowLib::setInput(float *input)
{
	_I1 = (_I1 == NULL) ? input : _I2;
	_I2 = input;
}


void FlowLib::getOutput(float *par_u1, float *par_u2)
{
	// TODO: Hier lieber memcpy verwenden
	for(unsigned int p=0;p<_nx*_ny;p++){
		par_u1[p] = _u1[p];
		par_u2[p] = _u2[p];
	}
}


const float *FlowLib::horizontal(){ return _u1;}
const float *FlowLib::vertical(){ return _u2;}

void FlowLib::setLambda(float par_lambda){_lambda = par_lambda;}
void FlowLib::setOverrelaxation(float par_overrelaxation){_overrelaxation = par_overrelaxation;}
void FlowLib::setOuterIterations(int par_oi){_oi = par_oi;}
void FlowLib::setInnerIterations(int par_ii){_ii = par_ii;}
void FlowLib::setStartLevel(int par_start_level){_start_level = par_start_level;}
void FlowLib::setEndLevel(int par_end_level){_end_level = par_end_level;}




/*########################## FlowLibSOR ################################*/
FlowLibSOR::FlowLibSOR(int par_nx, int par_ny):
FlowLib(par_nx,par_ny),
_dat_epsilon(SOR_DAT_EPSILON_INIT),
_reg_epsilon(SOR_REG_EPSILON_INIT),
_penDat(NULL), _penReg(NULL), _b1(NULL), _b2(NULL)
{
	fprintf(stderr,"\nCreating FlowLib SOR Size (%i|%i)",_nx,_ny);
	_overrelaxation = OVERRELAXATION_SOR_INIT;
}

FlowLibSOR::~FlowLibSOR()
{
}

int FlowLibSOR::computeMaxWarpLevels()
{
	return computeMaxWarpLevels(0.5f);
}

int FlowLibSOR::computeMaxWarpLevels // Bruhn Version with variable factor
(
		float rescale_factor
)
{
	int result;
	float nx,ny;

	for(result=1;;result++){
		nx=(int)ceil(_nx*pow(rescale_factor,result));
		ny=(int)ceil(_ny*pow(rescale_factor,result));

		if((nx<4)||(ny<4)) break;
	}

	if((nx==1)||(ny==1)) result--;
  return result;
}


FlowLibCpu::FlowLibCpu(int par_nx, int par_ny):
FlowLib(par_nx,par_ny),
_Ix(0),
_Iy(0),
_It(0),
_I1pyramid(0),
_I2pyramid(0)
{
	fprintf(stderr,"\nCreating FlowLib Cpu Size (%i|%i)",_nx,_ny);
	_Ix = new float[_nx*_ny];
	_Iy = new float[_nx*_ny];
	_It = new float[_nx*_ny];

	_I1pyramid = new ImagePyramidCpu(_nx,_ny);
	_I2pyramid = new ImagePyramidCpu(_nx,_ny);
}

FlowLibCpu::~FlowLibCpu()
{
	if(_I2warp) delete [] _I2warp;
	if(_u1lvl) delete [] _u1lvl;
	if(_u2lvl) delete [] _u2lvl;

	if(_Ix) delete [] _Ix;
	if(_Iy) delete [] _Iy;
	if(_It) delete [] _It;

	if(_I1pyramid) delete _I1pyramid;
	if(_I2pyramid) delete _I2pyramid;
}

void FlowLibCpu::setInput(float *input)
{
	bool fillI1pyramid = _I1pyramid->level[0] == NULL;
	FlowLib::setInput(input);

	ImagePyramidCpu *temp = _I1pyramid;
	_I1pyramid = _I2pyramid;
	_I2pyramid = temp;
	_I2pyramid->level[0] = _I2;
	_I2pyramid->fill();
	if(fillI1pyramid){
		_I1pyramid->level[0] = _I1;
		_I1pyramid->fill();
	}
}

//TODO : ZU einem Kernel zusammenfuegen, der die Posen von R,G,B parametrisiert
void FlowLibCpu::getOutputRGB
(
		float *output,
		float pmin,
		float pmax,
		float ptmin,
		float factor
)
{
	float temp;

	float Pi = 2.0f * acos(0.0f);

	float amp;
	float phi;
	float alpha, beta;

	for(unsigned int p=0;p<_nx*_ny;p++){

		float &r = output[3*p];
		float &g = output[3*p+1];
		float &b = output[3*p+2];

		/* determine amplitude and phase (cut amp at 1) */
		amp = (sqrtf(_u1[p]*_u1[p] + _u2[p]*_u2[p]) - pmin)/(pmax - pmin);
		if (amp > 1.0f) amp = 1.0f;
		if (_u1[p] == 0.0f)
			if (_u2[p] >= 0.0f) phi = 0.5f * Pi;
			else phi = 1.5f * Pi;
		else if (_u1[p] > 0.0f)
			if (_u2[p] >= 0.0f) phi = atanf(_u2[p]/_u1[p]);
			else phi = 2.0f * Pi + atanf(_u2[p]/_u1[p]);
		else phi = Pi + atanf (_u2[p]/_u1[p]);

		phi = phi / 2.0f;

		// interpolation between red (0) and blue (0.25 * Pi)
		if ((phi >= 0.0f) && (phi < 0.125f * Pi)) {
			beta  = phi / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha+ beta);
			g = 0.0f;
			b = amp * beta;
		}
		else if ((phi >= 0.125f * Pi) && (phi < 0.25f * Pi)) {
			beta  = (phi-0.125f * Pi) / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha + beta *  0.25f);
			g = amp * (beta *  0.25f);
			b = amp * (alpha + beta);
		}
		// interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
		else if ((phi >= 0.25f * Pi) && (phi < 0.375f * Pi)) {
			beta  = (phi - 0.25f * Pi) / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha *  0.25f);
			g = amp * (alpha *  0.25f + beta);
			b = amp * (alpha+ beta);
		}
		else if ((phi >= 0.375f * Pi) && (phi < 0.5f * Pi)) {
			beta  = (phi - 0.375f * Pi) / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = 0.0f;
			g = amp * (alpha+ beta);
			b = amp * (alpha);
		}
		// interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
		else if ((phi >= 0.5f * Pi) && (phi < 0.75f * Pi)) {
			beta  = (phi - 0.5f * Pi) / (0.25f * Pi);
			alpha = 1.0f - beta;
			r = amp * (beta);
			g = amp * (alpha+ beta);
			b = 0.0f;
		}
		// interpolation between yellow (0.75 * Pi) and red (Pi)
		else if ((phi >= 0.75f * Pi) && (phi <= Pi)) {
			beta  = (phi - 0.75f * Pi) / (0.25f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha+ beta);
			g = amp * (alpha);
			b = 0.0f;
		}
		else {
			r = g = b = 0.5f;
		}

		temp = atan2f(-_u1[p],-_u2[p])*0.954929659f; // 6/(2*PI)
		if (temp < 0) temp += 6.0f;

		r = (r>=1.0f)*1.0f+((r<1.0f)&&(r>0.0f))*r;
		g = (g>=1.0f)*1.0f+((g<1.0f)&&(g>0.0f))*g;
		b = (b>=1.0f)*1.0f+((b<1.0f)&&(b>0.0f))*b;

		r *= factor; g *= factor; b *= factor;
//		r = g = b = 0.5f;
	}
}



void FlowLibCpu::getOutputBGR
(
		float *output,
		float pmin,
		float pmax,
		float ptmin,
		float factor
)
{
	float temp;

	float Pi = 2.0f * acos(0.0f);

	float amp;
	float phi;
	float alpha, beta;

	for(unsigned int p=0;p<_nx*_ny;p++){

		float &r = output[3*p+2];
		float &g = output[3*p+1];
		float &b = output[3*p];

		/* determine amplitude and phase (cut amp at 1) */
		amp = (sqrtf(_u1[p]*_u1[p] + _u2[p]*_u2[p]) - pmin)/(pmax - pmin);
		if (amp > 1.0f) amp = 1.0f;
		if (_u1[p] == 0.0f)
			if (_u2[p] >= 0.0f) phi = 0.5f * Pi;
			else phi = 1.5f * Pi;
		else if (_u1[p] > 0.0f)
			if (_u2[p] >= 0.0f) phi = atanf(_u2[p]/_u1[p]);
			else phi = 2.0f * Pi + atanf(_u2[p]/_u1[p]);
		else phi = Pi + atanf (_u2[p]/_u1[p]);

		phi = phi / 2.0f;

		// interpolation between red (0) and blue (0.25 * Pi)
		if ((phi >= 0.0f) && (phi < 0.125f * Pi)) {
			beta  = phi / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha+ beta);
			g = 0.0f;
			b = amp * beta;
		}
		else if ((phi >= 0.125f * Pi) && (phi < 0.25f * Pi)) {
			beta  = (phi-0.125f * Pi) / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha + beta *  0.25f);
			g = amp * (beta *  0.25f);
			b = amp * (alpha + beta);
		}
		// interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
		else if ((phi >= 0.25f * Pi) && (phi < 0.375f * Pi)) {
			beta  = (phi - 0.25f * Pi) / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha *  0.25f);
			g = amp * (alpha *  0.25f + beta);
			b = amp * (alpha+ beta);
		}
		else if ((phi >= 0.375f * Pi) && (phi < 0.5f * Pi)) {
			beta  = (phi - 0.375f * Pi) / (0.125f * Pi);
			alpha = 1.0f - beta;
			r = 0.0f;
			g = amp * (alpha+ beta);
			b = amp * (alpha);
		}
		// interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
		else if ((phi >= 0.5f * Pi) && (phi < 0.75f * Pi)) {
			beta  = (phi - 0.5f * Pi) / (0.25f * Pi);
			alpha = 1.0f - beta;
			r = amp * (beta);
			g = amp * (alpha+ beta);
			b = 0.0f;
		}
		// interpolation between yellow (0.75 * Pi) and red (Pi)
		else if ((phi >= 0.75f * Pi) && (phi <= Pi)) {
			beta  = (phi - 0.75f * Pi) / (0.25f * Pi);
			alpha = 1.0f - beta;
			r = amp * (alpha+ beta);
			g = amp * (alpha);
			b = 0.0f;
		}
		else {
			r = g = b = 0.5f;
		}

		temp = atan2f(-_u1[p],-_u2[p])*0.954929659f; // 6/(2*PI)
		if (temp < 0) temp += 6.0f;

		r = (r>=1.0f)*1.0f+((r<1.0f)&&(r>0.0f))*r;
		g = (g>=1.0f)*1.0f+((g<1.0f)&&(g>0.0f))*g;
		b = (b>=1.0f)*1.0f+((b<1.0f)&&(b>0.0f))*b;

		r *= factor; g *= factor; b *= factor;

	}
}













