/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    flowlib_cpu_sor.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * flowlib_cpu_sor.cpp
 *
 *  Created on: Mar 14, 2012
 *      Author: steinbrf
 */


//#include <flowlib_cpu_sor.hpp>
#include "flowlib.hpp"
#include <stdlib.h>
#include <linearoperations/linearoperations.h>
#include <math.h>
#include <stdio.h>
#include <auxiliary/debug.hpp>

FlowLibCpuSOR::FlowLibCpuSOR(int par_nx, int par_ny):
FlowLib(par_nx,par_ny),FlowLibCpu(par_nx,par_ny),FlowLibSOR(par_nx,par_ny)
{
	_penDat = new float[_nx*_ny];
	_penReg = new float[_nx*_ny];
	_b1 = new float[_nx*_ny];
	_b2 = new float[_nx*_ny];
}

FlowLibCpuSOR::~FlowLibCpuSOR()
{
	if(_penDat) delete [] _penDat;
	if(_penReg) delete [] _penReg;
	if(_b1)     delete [] _b1;
	if(_b2)     delete [] _b2;
}



float FlowLibCpuSOR::computeFlow()
{
	float lambda = _lambda * 255.0f;

	int   max_rec_depth;
	int   warp_max_levels;
	int   rec_depth;

	warp_max_levels = computeMaxWarpLevels();

  max_rec_depth = (((_start_level+1) < warp_max_levels) ?
                    (_start_level+1) : warp_max_levels) -1;

  if(max_rec_depth >= _I1pyramid->nl){
  	max_rec_depth = _I1pyramid->nl-1;
  }

	if(_verbose) fprintf(stderr,"\nFlow GPU SOR Relaxation: %f",_overrelaxation);

  unsigned int nx_fine, ny_fine, nx_coarse=0, ny_coarse=0;

	float hx_fine;
	float hy_fine;

  for(unsigned int p=0;p<_nx*_ny;p++){
  	_u1[p] = _u2[p] = 0.0f;
  }

	for(rec_depth = max_rec_depth; rec_depth >= 0; rec_depth--)	{

		if(_verbose) fprintf(stderr," Level %i",rec_depth);

		nx_fine = _I1pyramid->nx[rec_depth];
		ny_fine = _I1pyramid->ny[rec_depth];

		hx_fine=(float)_nx/(float)nx_fine;
		hy_fine=(float)_ny/(float)ny_fine;

		const float hx_1 = 1.0f / (2.0f*hx_fine);
		const float hy_1 = 1.0f / (2.0f*hy_fine);
		const float hx_2 = lambda/(hx_fine*hx_fine);
		const float hy_2 = lambda/(hy_fine*hy_fine);

		if(_debug){
			sprintf(_debugbuffer,"debug/CI1 %i.png",rec_depth);
			saveFloatImage(_debugbuffer,_I1pyramid->level[rec_depth],nx_fine,ny_fine,1,1.0f,-1.0f);
			sprintf(_debugbuffer,"debug/CI2 %i.png",rec_depth);
			saveFloatImage(_debugbuffer,_I2pyramid->level[rec_depth],nx_fine,ny_fine,1,1.0f,-1.0f);
		}

		if(rec_depth < max_rec_depth)	{
			resampleAreaParallelizableSeparate(_u1,_u1,nx_coarse,ny_coarse,nx_fine,ny_fine,_b1);
			resampleAreaParallelizableSeparate(_u2,_u2,nx_coarse,ny_coarse,nx_fine,ny_fine,_b2);
		}

		if(rec_depth >= _end_level){
			backwardRegistrationBilinearFunction(_I2pyramid->level[rec_depth],_I2warp,
					_u1,_u2,_I1pyramid->level[rec_depth],
					nx_fine,ny_fine,hx_fine,hy_fine);

			if(_debug){
				sprintf(_debugbuffer,"debug/CW2 %i.png",rec_depth);
				saveFloatImage(_debugbuffer,_I2warp,nx_fine,ny_fine,1,1.0f,-1.0f);
			}

			for(unsigned int p=0;p<nx_fine*ny_fine;p++) _u1lvl[p] = _u2lvl[p] = 0.0f;

			for(unsigned int i=0;i<_oi;i++){

				//Update Robustifications
				for(unsigned int x=0;x<nx_fine;x++){
					unsigned int x_1 = x==0     ? x : x-1;
					unsigned int x1 = x==nx_fine-1 ? x : x+1;
					for(unsigned int y=0;y<ny_fine;y++){
						unsigned int y_1 = y==0     ? y : y-1;
						unsigned int y1 = y==ny_fine-1 ? y : y+1;

						float Ix = 0.5f*(_I2warp[y*nx_fine+x1]-_I2warp[y*nx_fine+x_1] +
								_I1pyramid->level[rec_depth][y*nx_fine+x1]-_I1pyramid->level[rec_depth][y*nx_fine+x_1])*hx_1;
						float Iy = 0.5f*(_I2warp[y1*nx_fine+x]-_I2warp[y_1*nx_fine+x] +
								_I1pyramid->level[rec_depth][y1*nx_fine+x]-_I1pyramid->level[rec_depth][y_1*nx_fine+x])*hy_1;
						float It = _I2warp[y*nx_fine+x] - _I1pyramid->level[rec_depth][y*nx_fine+x];

						double dxu = (_u1[y*nx_fine+x1] - _u1[y*nx_fine+x_1] + _u1lvl[y*nx_fine+x1] - _u1lvl[y*nx_fine+x_1])*hx_1;
						double dyu = (_u1[y1*nx_fine+x] - _u1[y_1*nx_fine+x] + _u1lvl[y1*nx_fine+x] - _u1lvl[y_1*nx_fine+x])*hy_1;
						double dxv = (_u2[y*nx_fine+x1] - _u2[y*nx_fine+x_1] + _u2lvl[y*nx_fine+x1] - _u2lvl[y*nx_fine+x_1])*hx_1;
						double dyv = (_u2[y1*nx_fine+x] - _u2[y_1*nx_fine+x] + _u2lvl[y1*nx_fine+x] - _u2lvl[y_1*nx_fine+x])*hy_1;

						double dataterm = _u1lvl[y*nx_fine+x]*Ix + _u2lvl[y*nx_fine+x]*Iy + It;
						_penDat[y*nx_fine+x] = 1.0f / sqrt(dataterm*dataterm + _dat_epsilon);
						_penReg[y*nx_fine+x] = 1.0f / sqrt(dxu*dxu + dxv*dxv + dyu*dyu + dyv*dyv + _reg_epsilon);

					}
				}

				//Update Righthand Side
				for(unsigned int x=0;x<nx_fine;x++){
					unsigned int x_1 = x==0     ? x : x-1;
					unsigned int x1 = x==nx_fine-1 ? x : x+1;
					for(unsigned int y=0;y<ny_fine;y++){
						unsigned int y_1 = y==0     ? y : y-1;
						unsigned int y1 = y==ny_fine-1 ? y : y+1;

						float Ix = 0.5f*(_I2warp[y*nx_fine+x1]-_I2warp[y*nx_fine+x_1] +
								_I1pyramid->level[rec_depth][y*nx_fine+x1]-_I1pyramid->level[rec_depth][y*nx_fine+x_1])*hx_1;
						float Iy = 0.5f*(_I2warp[y1*nx_fine+x]-_I2warp[y_1*nx_fine+x] +
								_I1pyramid->level[rec_depth][y1*nx_fine+x]-_I1pyramid->level[rec_depth][y_1*nx_fine+x])*hy_1;
						float It = _I2warp[y*nx_fine+x] - _I1pyramid->level[rec_depth][y*nx_fine+x];

						float xp = x<nx_fine-1 ? (_penReg[y*nx_fine+x1] +_penReg[y*nx_fine+x])*0.5f*hx_2 : 0.0f;
						float xm = x>0         ? (_penReg[y*nx_fine+x_1]+_penReg[y*nx_fine+x])*0.5f*hx_2 : 0.0f;
						float yp = y<ny_fine-1 ? (_penReg[y1*nx_fine+x] +_penReg[y*nx_fine+x])*0.5f*hy_2 : 0.0f;
						float ym = y>0         ? (_penReg[y_1*nx_fine+x]+_penReg[y*nx_fine+x])*0.5f*hy_2 : 0.0f;
						float sum = xp + xm + yp + ym;

						_b1[y*nx_fine+x] = -_penDat[y*nx_fine+x] * Ix*It
								+ (x>0     ? xm*_u1[y*nx_fine+x_1] : 0.0f)
								+ (x<nx_fine-1 ? xp*_u1[y*nx_fine+x1] : 0.0f)
								+ (y>0     ? ym*_u1[y_1*nx_fine+x] : 0.0f)
								+ (y<ny_fine-1 ? yp*_u1[y1*nx_fine+x] : 0.0f)
								- sum * _u1[y*nx_fine+x];

						_b2[y*nx_fine+x] = -_penDat[y*nx_fine+x] * Iy*It
								+ (x>0     ? xm*_u2[y*nx_fine+x_1] : 0.0f)
								+ (x<nx_fine-1 ? xp*_u2[y*nx_fine+x1] : 0.0f)
								+ (y>0     ? ym*_u2[y_1*nx_fine+x] : 0.0f)
								+ (y<ny_fine-1 ? yp*_u2[y1*nx_fine+x] : 0.0f)
								- sum * _u2[y*nx_fine+x];
					}
				}

				for(unsigned int j=0;j<_ii;j++){

					for(unsigned int x=0;x<nx_fine;x++){
						unsigned int x_1 = x==0     ? x : x-1;
						unsigned int x1 = x==nx_fine-1 ? x : x+1;
						for(unsigned int y=0;y<ny_fine;y++){
							unsigned int y_1 = y==0     ? y : y-1;
							unsigned int y1 = y==ny_fine-1 ? y : y+1;

							unsigned int p = y*nx_fine+x;

							float Ix = 0.5f*(_I2warp[y*nx_fine+x1]-_I2warp[y*nx_fine+x_1] +
									_I1pyramid->level[rec_depth][y*nx_fine+x1]-_I1pyramid->level[rec_depth][y*nx_fine+x_1])*hx_1;
							float Iy = 0.5f*(_I2warp[y1*nx_fine+x]-_I2warp[y_1*nx_fine+x] +
									_I1pyramid->level[rec_depth][y1*nx_fine+x]-_I1pyramid->level[rec_depth][y_1*nx_fine+x])*hy_1;

							float xp = x<nx_fine-1 ? (_penReg[y*nx_fine+x1] +_penReg[y*nx_fine+x])*0.5f*hx_2 : 0.0f;
							float xm = x>0         ? (_penReg[y*nx_fine+x_1]+_penReg[y*nx_fine+x])*0.5f*hx_2 : 0.0f;
							float yp = y<ny_fine-1 ? (_penReg[y1*nx_fine+x]+_penReg[y*nx_fine+x])*0.5f*hy_2  : 0.0f;
							float ym = y>0         ? (_penReg[y_1*nx_fine+x]+_penReg[y*nx_fine+x])*0.5f*hy_2 : 0.0f;
							float sum = xp + xm + yp + ym;

							if((x+y)%2==0){
								float u1new  = (1.0f-_overrelaxation)*_u1lvl[p] + _overrelaxation *
										(_b1[p] - _penDat[p] * Ix*Iy * _u2lvl[p]
										+ (x>0         ? xm*_u1lvl[y*nx_fine+x_1] : 0.0f)
										+ (x<nx_fine-1 ? xp*_u1lvl[y*nx_fine+x1]  : 0.0f)
										+ (y>0         ? ym*_u1lvl[y_1*nx_fine+x] : 0.0f)
										+ (y<ny_fine-1 ? yp*_u1lvl[y1*nx_fine+x]  : 0.0f)
										) / (_penDat[p] * Ix*Ix + sum);

								float u2new = (1.0f-_overrelaxation)*_u2lvl[p] + _overrelaxation *
										(_b2[p] - _penDat[p] * Ix*Iy * _u1lvl[p]
										+ (x>0         ? xm*_u2lvl[y*nx_fine+x_1] : 0.0f)
										+ (x<nx_fine-1 ? xp*_u2lvl[y*nx_fine+x1]  : 0.0f)
										+ (y>0         ? ym*_u2lvl[y_1*nx_fine+x] : 0.0f)
										+ (y<ny_fine-1 ? yp*_u2lvl[y1*nx_fine+x]  : 0.0f))
										/ (_penDat[p] * Iy*Iy + sum);
								_u1lvl[p] = u1new;
								_u2lvl[p] = u2new;
							}
						}
					}
					for(unsigned int x=0;x<nx_fine;x++){
						unsigned int x_1 = x==0     ? x : x-1;
						unsigned int x1 = x==nx_fine-1 ? x : x+1;
						for(unsigned int y=0;y<ny_fine;y++){
							unsigned int y_1 = y==0     ? y : y-1;
							unsigned int y1 = y==ny_fine-1 ? y : y+1;

							unsigned int p = y*nx_fine+x;

							float Ix = 0.5f*(_I2warp[y*nx_fine+x1]-_I2warp[y*nx_fine+x_1] +
									_I1pyramid->level[rec_depth][y*nx_fine+x1]-_I1pyramid->level[rec_depth][y*nx_fine+x_1])*hx_1;
							float Iy = 0.5f*(_I2warp[y1*nx_fine+x]-_I2warp[y_1*nx_fine+x] +
									_I1pyramid->level[rec_depth][y1*nx_fine+x]-_I1pyramid->level[rec_depth][y_1*nx_fine+x])*hy_1;

							float xp = x<nx_fine-1 ? (_penReg[y*nx_fine+x1] +_penReg[y*nx_fine+x])*0.5f*hx_2 : 0.0f;
							float xm = x>0         ? (_penReg[y*nx_fine+x_1]+_penReg[y*nx_fine+x])*0.5f*hx_2 : 0.0f;
							float yp = y<ny_fine-1 ? (_penReg[y1*nx_fine+x] +_penReg[y*nx_fine+x])*0.5f*hy_2 : 0.0f;
							float ym = y>0         ? (_penReg[y_1*nx_fine+x]+_penReg[y*nx_fine+x])*0.5f*hy_2 : 0.0f;
							float sum = xp + xm + yp + ym;

							if((x+y)%2==1){
								float u1new  = (1.0f-_overrelaxation)*_u1lvl[p] + _overrelaxation *
										(_b1[p] - _penDat[p] * Ix*Iy * _u2lvl[p]
										+ (x>0         ? xm*_u1lvl[y*nx_fine+x_1] : 0.0f)
										+ (x<nx_fine-1 ? xp*_u1lvl[y*nx_fine+x1]  : 0.0f)
										+ (y>0         ? ym*_u1lvl[y_1*nx_fine+x] : 0.0f)
										+ (y<ny_fine-1 ? yp*_u1lvl[y1*nx_fine+x]  : 0.0f)
										) / (_penDat[p] * Ix*Ix + sum);

								float u2new = (1.0f-_overrelaxation)*_u2lvl[p] + _overrelaxation *
										(_b2[p] - _penDat[p] * Ix*Iy * _u1lvl[p]
										+ (x>0         ? xm*_u2lvl[y*nx_fine+x_1] : 0.0f)
										+ (x<nx_fine-1 ? xp*_u2lvl[y*nx_fine+x1]  : 0.0f)
										+ (y>0         ? ym*_u2lvl[y_1*nx_fine+x] : 0.0f)
										+ (y<ny_fine-1 ? yp*_u2lvl[y1*nx_fine+x]  : 0.0f))
										/ (_penDat[p] * Iy*Iy + sum);
								_u1lvl[p] = u1new;
								_u2lvl[p] = u2new;
							}
						}
					}
				}
			}

			for(unsigned int p=0;p<nx_fine*ny_fine;p++){
				_u1[p] += _u1lvl[p];
				_u2[p] += _u2lvl[p];
			}
		}
		else{
			if(_verbose) fprintf(stderr," skipped");
		}
		nx_coarse = nx_fine;
		ny_coarse = ny_fine;
	}

	if(_debug) delete [] _debugbuffer;

	//TODO: Timer
	return -1.0f;
}


