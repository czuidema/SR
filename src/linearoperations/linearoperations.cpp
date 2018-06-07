/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    linearoperations.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * blur_resample.cpp
 *
 *  Created on: Mar 15, 2012
 *      Author: steinbrf
 */
#include "linearoperations.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

#ifndef RESAMPLING_INTERPOLATION_OFFSET
#define RESAMPLING_INTERPOLATION_OFFSET -0.5f
#endif

#ifndef RESAMPLE_EPSILON
#define RESAMPLE_EPSILON 0.005f
#endif


void resampleAreaParallelizableSeparate_x
(
	const float *in,
	float *out,
	int   nx,
	int   ny,
	float hx,
	int   nx_orig,
	float factor = 0.0f
)
{

	if(factor == 0.0f) factor = 1.0f/hx;

	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			int p = y*nx+x;

			float px = (float)x * hx;
			float left = ceil(px) - px;
			if(left > hx) left = hx;
			float midx = hx - left;
			float right = midx - floorf(midx);
			midx = midx - right;

			out[p] = 0.0f;

			if(left > 0.0f){
				out[p] += in[y*nx_orig+(int)(floor(px))]*left*factor;
				px+= 1.0f;
			}
			while(midx > 0.0f){
				out[p] += in[y*nx_orig+(int)(floor(px))]*factor;
				px += 1.0f;
				midx -= 1.0f;
			}
			if(right > RESAMPLE_EPSILON)	{
				out[p] += in[y*nx_orig+(int)(floor(px))]*right*factor;
			}
		}
	}
}


void resampleAreaParallelizableSeparate_y
(
	const float *in,
	float *out,
	int   nx,
	int   ny,
	float hy,
	float factor = 0.0f
)
{
	if(factor == 0.0f) factor = 1.0f/hy;

	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			int p = y*nx+x;
			float py = (float)y * hy;
			float top = ceil(py) - py;
			if(top > hy) top = hy;
			float midy = hy - top;
			float bottom = midy - floorf(midy);
			midy = midy - bottom;

			out[p] = 0.0f;

			if(top > 0.0f){
				out[p] += in[(int)(floor(py))*nx+x]*top*factor;
				py += 1.0f;
			}
			while(midy > 0.0f){
				out[p] += in[(int)(floor(py))*nx+x]*factor;
				py += 1.0f;
				midy -= 1.0f;
			}
			if(bottom > RESAMPLE_EPSILON){
				out[p] += in[(int)(floor(py))*nx+x]*bottom*factor;
			}
		}
	}
}



void resampleAreaParallelizableSeparate
(
	const float *in,
	float *out,
	int   nx_in,
	int   ny_in,
	int   nx_out,
	int   ny_out,
	float *help,
	float scalefactor
)
{
	bool selfalloc = help == 0;
	if(selfalloc){
		fprintf(stderr,"\nADVICE: Use a helper array for separate Resampling!");
		help = new float[std::max(nx_in,nx_out)*std::max(ny_in,ny_out)];
	}
	resampleAreaParallelizableSeparate_x(in,help,nx_out,ny_in,
			(float)(nx_in)/(float)(nx_out),nx_in,(float)(nx_out)/(float)(nx_in));
	resampleAreaParallelizableSeparate_y(help,out,nx_out,ny_out,
			(float)(ny_in)/(float)(ny_out),scalefactor*(float)(ny_out)/(float)(ny_in));
	if(selfalloc){
		delete [] help;
	}
}


void resampleAreaParallelizableSeparateAdjoined
(
	const float *in,
	float *out,
	int   nx_in,
	int   ny_in,
	int   nx_out,
	int   ny_out,
	float *help,
	float scalefactor
)
{
	bool selfalloc = help == 0;
	if(selfalloc){
		fprintf(stderr,"\nADVICE: Use a helper array for separate Resampling!");
		help = new float[std::max(nx_in,nx_out)*std::max(ny_in,ny_out)];
	}
	resampleAreaParallelizableSeparate_x(in,help,nx_out,ny_in,(float)(nx_in)/(float)(nx_out),nx_in,1.0f);
	resampleAreaParallelizableSeparate_y(help,out,nx_out,ny_out,(float)(ny_in)/(float)(ny_out),scalefactor);
	if(selfalloc){
		delete [] help;
	}
}

void backwardRegistrationBilinearValue
(
 const float *f2,
 float *f2_bw,
 const float *u,
 const float *v,
 float value,
 int   nx,
 int   ny,
 float hx,
 float hy
)
{
	float hx_1 = 1.0f/hx;
	float hy_1 = 1.0f/hy;

	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			float ii_fp = x+(u[y*nx+x]*hx_1);
			float jj_fp = y+(v[y*nx+x]*hy_1);

			if((ii_fp < 0.0f) || (jj_fp < 0.0f)
						 || (ii_fp > (float)(nx-1)) || (jj_fp > (float)(ny-1))){
				f2_bw[y*nx+x] = value;
			}
			else if(!isfinite(ii_fp) || !isfinite(jj_fp)){
				fprintf(stderr,"!");
				f2_bw[y*nx+x] = value;
			}
			else{
				int xx = (int)floor(ii_fp);
				int yy = (int)floor(jj_fp);

				int xx1 = xx == nx-1 ? xx : xx+1;
				int yy1 = yy == ny-1 ? yy : yy+1;

				float xx_rest = ii_fp - (float)xx;
				float yy_rest = jj_fp - (float)yy;

				f2_bw[y*nx+x] = (1.0f-xx_rest)*(1.0f-yy_rest) * f2[yy*nx+xx]
						+ xx_rest*(1.0f-yy_rest) * f2[yy*nx+xx1]
						+ (1.0f-xx_rest)*yy_rest * f2[(yy1)*nx+xx]
						+ xx_rest * yy_rest      * f2[(yy1)*nx+xx1];
			}
		}
	}
}

void backwardRegistrationBilinearFunction
(
 const float *I2,
 float *I2warp,
 const float *flow1,
 const float *flow2,
 const float *constant,
 int   nx,
 int   ny,
 float hx,
 float hy
)
{
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			const float xx = (float)x+flow1[y*nx+x]/hx;
			const float yy = (float)y+flow2[y*nx+x]/hy;

      int xxFloor = (int)floor(xx);
      int yyFloor = (int)floor(yy);

      int xxCeil = xxFloor == nx-1 ? xxFloor : xxFloor+1;
      int yyCeil = yyFloor == ny-1 ? yyFloor : yyFloor+1;

      float xxRest = xx - (float)xxFloor;
      float yyRest = yy - (float)yyFloor;

      I2warp[y*nx+x] =
          (xx < 0.0f || yy < 0.0f || xx > (float)(nx-1) || yy > (float)(ny-1))
          ? constant[y*nx+x] :
        (1.0f-xxRest)*(1.0f-yyRest) * I2[yyFloor*nx+xxFloor]
            + xxRest*(1.0f-yyRest)  * I2[yyFloor*nx+xxCeil]
            + (1.0f-xxRest)*yyRest  * I2[yyCeil*nx+xxFloor]
            + xxRest * yyRest       * I2[yyCeil*nx+xxCeil];
		}
	}
}

void forewardRegistrationBilinear
(
	float *in,
	float *out,
	float *u,
	float *v,
	int   nx,
	int   ny
)
{
	for(int x=0;x<nx*ny;x++){
		out[x] = 0.0f;
	}

	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			const int p = y*nx+x;
			const float xx = (float)x + u[p];
			const float yy = (float)y + v[p];
			if(xx >= 0.0f && xx <= (float)(nx-2) && yy >= 0.0f && yy <= (float)(ny-2))
			{
				float xxf = floor(xx);
				float yyf = floor(yy);
				const int xxi = (int)xxf;
				const int yyi = (int)yyf;
				xxf = xx - xxf;
				yyf = yy - yyf;

				out[yyi*nx+xxi] += in[p] * (1.0f-xxf)*(1.0f-yyf);
				out[yyi*nx+xxi+1] += in[p] * xxf*(1.0f-yyf);
				out[(yyi+1)*nx+xxi] += in[p] * (1.0f-xxf)*yyf;
				out[(yyi+1)*nx+xxi+1] += in[p] * xxf*yyf;
			}
		}
	}
}


void gaussBlurSeparateMirror
(
	float *in,
	float *out,
	int   nx,
	int   ny,
	float sigmax,
	float sigmay,
	int   radius,
	float *temp,
	float *mask
)
{
	if(sigmax <= 0.0f || sigmay <= 0.0f || radius < 0){
		return;
	}

	if(radius == 0){
		int maxsigma = (sigmax > sigmay) ? sigmax : sigmay;
		radius = (int)(3.0f*maxsigma);
	}

	bool selfalloctemp = temp == NULL;
	if(selfalloctemp) temp = new float[nx*ny];
	bool selfallocmask = mask == NULL;
	if(selfallocmask) mask = new float[radius+1];

	float result, sum;
	sigmax = 1.0f/(sigmax*sigmax);
	sigmay = 1.0f/(sigmay*sigmay);

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmax));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++){
			result = mask[0]*in[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((x-i>=0) ? in[y*nx+(x-i)] : in[y*nx+(-1-(x-i))]) +
						((x+i<nx) ? in[y*nx+(x+i)] : in[y*nx+(nx-(x+i-(nx-1)))]));
			}
			temp[y*nx+x] = result;
		}
	}

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmay));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	for(int x=0;x<nx;x++){
		for(int y=0;y<ny;y++)	{
			result = mask[0]*temp[y*nx+x];
			for(int i=1;i<=radius;i++){
				result += mask[i]*(
						((y-i>=0) ? temp[(y-i)*nx+x] : temp[(-1-(y-i))*nx+x]) +
						((y+i<ny) ? temp[(y+i)*nx+x] : temp[(ny-(y+i-(ny-1)))*nx+x]));
			}
			out[y*nx+x] = result;
		}
	}

	if(selfallocmask) delete [] mask;
	if(selfalloctemp) delete [] temp;
}





