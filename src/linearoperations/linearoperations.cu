/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    linearoperations.cu
*
*
* implement all functions with ### implement me ### in the function body
\****************************************************************************/ 

/*
 * linearoperations.cu
 *
 *  Created on: Aug 3, 2012
 *      Author: steinbrf
 */


#include <auxiliary/cuda_basic.cuh>

cudaChannelFormatDesc linearoperation_float_tex = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_linearoperation;
bool linearoperation_textures_initialized = false;


#define MAXKERNELRADIUS     20    // maximum allowed kernel radius
#define MAXKERNELSIZE   21    // maximum allowed kernel radius + 1
__constant__ float constKernel[MAXKERNELSIZE];


void setTexturesLinearOperations(int mode){
	tex_linearoperation.addressMode[0] = cudaAddressModeClamp;
	tex_linearoperation.addressMode[1] = cudaAddressModeClamp;
	if(mode == 0)tex_linearoperation.filterMode = cudaFilterModePoint;
	else tex_linearoperation.filterMode = cudaFilterModeLinear;
	tex_linearoperation.normalized = false;
}


#define LO_TEXTURE_OFFSET 0.5f
#define LO_RS_AREA_OFFSET 0.0f

#ifdef DGT400
#define LO_BW 32
#define LO_BH 16
#else
#define LO_BW 16
#define LO_BH 16
#endif


#ifndef RESAMPLE_EPSILON
#define RESAMPLE_EPSILON 0.005f
#endif

#ifndef atomicAdd
__device__ float atomicAdd(float* address, double val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__float_as_int(val + __int_as_float(assumed)));
	}	while (assumed != old);
	return __int_as_float(old);
}

#endif

__global__ void backwardRegistrationValue_tex_kernel
(
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		float value,
		int   nx,
		int   ny,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1_out+x;

	if(x<nx && y<ny){
		float xx = (float)x+flow1_g[pg]/hx;
		float yy = (float)y+flow2_g[pg]/hy;
		out_g[pg] = (xx < 0.0f || yy < 0.0f || xx > (float)(nx-1) || yy > (float)(ny-1))
				? 0.0f : tex2D(tex_linearoperation,xx+LO_TEXTURE_OFFSET,yy+LO_TEXTURE_OFFSET);
	}
}

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
		float hx,
		float hy
)
{
	dim3 dimBlock(LO_BW,LO_BH);
	dim3 dimGrid((nx%LO_BW) ? (nx/LO_BW+1) : (nx/LO_BW),(ny%LO_BH) ? (ny/LO_BH+1) : (ny/LO_BH));

	setTexturesLinearOperations(1);
	catchkernel;
	cutilSafeCall(cudaBindTexture2D(0,&tex_linearoperation,in_g,&linearoperation_float_tex,nx,ny,pitchf1_in*sizeof(float)));
	catchkernel;
	backwardRegistrationValue_tex_kernel<<<dimGrid,dimBlock>>>(flow1_g,flow2_g,out_g,value,nx,ny,pitchf1_out,hx,hy);
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));
}

__global__ void backwardRegistrationFunction_global_kernel
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
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1_out+x;

	const float xx = (float)x+flow1_g[pg]/hx;
	const float yy = (float)y+flow2_g[pg]/hy;

  int xxFloor = (int)floor(xx);
  int yyFloor = (int)floor(yy);

  int xxCeil = xxFloor == nx-1 ? xxFloor : xxFloor+1;
  int yyCeil = yyFloor == ny-1 ? yyFloor : yyFloor+1;

  float xxRest = xx - (float)xxFloor;
  float yyRest = yy - (float)yyFloor;

	if(x<nx && y<ny){
		out_g[pg] =
				(xx < 0.0f || yy < 0.0f || xx > (float)(nx-1) || yy > (float)(ny-1))
				? constant_g[y*pitchf1_in+x] :
		        (1.0f-xxRest)*(1.0f-yyRest) * in_g[yyFloor*pitchf1_in+xxFloor]
		            + xxRest*(1.0f-yyRest)  * in_g[yyFloor*pitchf1_in+xxCeil]
		            + (1.0f-xxRest)*yyRest  * in_g[yyCeil*pitchf1_in+xxFloor]
		            + xxRest * yyRest       * in_g[yyCeil*pitchf1_in+xxCeil];
	}
}

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
)
{
	dim3 dimBlock(LO_BW,LO_BH);
	dim3 dimGrid((nx%LO_BW) ? (nx/LO_BW+1) : (nx/LO_BW),(ny%LO_BH) ? (ny/LO_BH+1) : (ny/LO_BH));

	backwardRegistrationFunction_global_kernel<<<dimGrid,dimBlock>>>
			(in_g,flow1_g,flow2_g,out_g,constant_g,nx,ny,pitchf1_in,pitchf1_out,hx,hy);
	catchkernel;
}

__global__ void backwardRegistrationFunction_tex_kernel
(
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
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1_out+x;

	const float xx = (float)x+flow1_g[pg]/hx;
	const float yy = (float)y+flow2_g[pg]/hy;

	if(x<nx && y<ny){
		out_g[pg] =
				(xx < 0.0f || yy < 0.0f || xx > (float)(nx-1) || yy > (float)(ny-1))
				? constant_g[y*pitchf1_in+x] :
				 tex2D(tex_linearoperation,xx+LO_TEXTURE_OFFSET,yy+LO_TEXTURE_OFFSET);
	}
}

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
		float hx,
		float hy
)
{
	dim3 dimBlock(LO_BW,LO_BH);
	dim3 dimGrid((nx%LO_BW) ? (nx/LO_BW+1) : (nx/LO_BW),(ny%LO_BH) ? (ny/LO_BH+1) : (ny/LO_BH));

	setTexturesLinearOperations(1);
	cutilSafeCall(cudaBindTexture2D(0,&tex_linearoperation,in_g,&linearoperation_float_tex,nx,ny,pitchf1_in*sizeof(float)));
	backwardRegistrationFunction_tex_kernel<<<dimGrid,dimBlock>>>(flow1_g,flow2_g,out_g,constant_g,nx,ny,pitchf1_in,pitchf1_out,hx,hy);
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));
}


__global__ void forewardRegistrationAtomic_Kernel
(
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
	  float       *out_g,
		int         nx,
		int         ny,
		int         pitchf1
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;

	if(x<nx && y<ny){
		float in = in_g[pg];
		float xx = (float)x+flow1_g[pg];
		float yy = (float)y+flow2_g[pg];
		if(xx >= 0.0f && xx <= (float)(nx-2) && yy >= 0.0f && yy <= (float)(ny-2))
		{
			float xxf = floor(xx);
			float yyf = floor(yy);
			int xxi = (int)xxf;
			int yyi = (int)yyf;
			xxf = xx - xxf;
			yyf = yy - yyf;

			atomicAdd(out_g + yyi*pitchf1+xxi      ,in * (1.0f-xxf)*(1.0f-yyf));
			atomicAdd(out_g + yyi*pitchf1+xxi+1    ,in * xxf*(1.0f-yyf));
			atomicAdd(out_g + (yyi+1)*pitchf1+xxi  ,in * (1.0f-xxf)*yyf);
			atomicAdd(out_g + (yyi+1)*pitchf1+xxi+1,in * xxf*yyf);

		}
	}
}

void forewardRegistrationBilinearAtomic
(
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
	  float       *out_g,
		int         nx,
		int         ny,
		int         pitchf1
)
{
	dim3 dimBlock(LO_BW,LO_BH);
	dim3 dimGrid((nx%LO_BW) ? (nx/LO_BW+1) : (nx/LO_BW),(ny%LO_BH) ? (ny/LO_BH+1) : (ny/LO_BH));
	forewardRegistrationAtomic_Kernel<<<dimGrid,dimBlock>>>(flow1_g,flow2_g,in_g,out_g,nx,ny,pitchf1);
	catchkernel;
}


__global__ void gaussBlurSeparateMirrorX_const_tex
(
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		int   radius
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	if(x<nx && y<ny){
		const float xx = (float)(x) + LO_TEXTURE_OFFSET;
		const float yy = (float)(y) + LO_TEXTURE_OFFSET;
		float result = tex2D(tex_linearoperation,xx,yy) * constKernel[0];
		for(int i=1;i<=radius;i++){
			//TODO: Klammern rausnehmen
			result += constKernel[i] *
					(tex2D(tex_linearoperation,(x-i>=0) ? (xx-i) : (-1.0f-(xx-i)),yy) +
					 tex2D(tex_linearoperation,(x+i<nx) ? (xx+i) : (nx-(xx+i-(nx-1.0f))),yy))	;
			out_g[pg] = result;
		}
	}
}

__global__ void gaussBlurSeparateMirrorY_const_tex
(
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		int   radius
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	if(x<nx && y<ny){
		const float xx = (float)(x) + LO_TEXTURE_OFFSET;
		const float yy = (float)(y) + LO_TEXTURE_OFFSET;
		float result = tex2D(tex_linearoperation,xx,yy) * constKernel[0];
		for(int i=1;i<=radius;i++){
			//TODO: Klammern rausnehmen
			result += constKernel[i] *
					(tex2D(tex_linearoperation,xx,(y-i>=0) ? (yy-i) :(-1.0f-(yy-i))) +
					 tex2D(tex_linearoperation,xx,(y+i<ny) ? (yy+i) : (ny-(yy+i-(ny-1.0f)))))	;
			out_g[pg] = result;
		}
	}
}

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
)
{
	if(sigmax <= 0.0f || sigmay <= 0.0f || radius <= 0)
	{
		return;
	}
	bool selfalloctemp = temp_g == NULL;
	if(selfalloctemp) cuda_malloc2D((void**)&temp_g,nx,ny,1,sizeof(float),&pitchf1);
	bool selfallocmask = mask == NULL;
	if(selfallocmask) mask = new float[radius+1];

	setTexturesLinearOperations(1);

	float sum;
	sigmax = 1.0f/(sigmax*sigmax);
	sigmay = 1.0f/(sigmay*sigmay);

	dim3 dimBlock(LO_BW,LO_BH);
	dim3 dimGrid((nx%LO_BW) ? (nx/LO_BW+1) : (nx/LO_BW),(ny%LO_BH) ? (ny/LO_BH+1) : (ny/LO_BH));

	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmax));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
	//FIXME: Was ist an der ersten Version falsch??
//  cutilSafeCall( cudaMemcpyToSymbol( (const char *)constKernel, mask, (radius+1)*sizeof(float)) );
  cutilSafeCall( cudaMemcpyToSymbol(constKernel, mask, (radius+1)*sizeof(float)) );
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, in_g,
		&linearoperation_float_tex, nx , ny, pitchf1*sizeof(float)) );
	gaussBlurSeparateMirrorX_const_tex<<<dimGrid,dimBlock>>>(temp_g,nx,ny,pitchf1,radius);
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));


	mask[0] = sum = 1.0f;
	for(int x=1;x<=radius;x++)	{
		mask[x] = exp(-0.5f*((float)(x*x)*sigmay));
		sum += 2.0f*mask[x];
	}
	for(int x=0;x<=radius;x++)	{
		mask[x] /= sum;
	}
//  cutilSafeCall( cudaMemcpyToSymbol( (const char *)constKernel, mask, (radius+1)*sizeof(float)) );
  cutilSafeCall( cudaMemcpyToSymbol(constKernel, mask, (radius+1)*sizeof(float)) );
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, temp_g,
		&linearoperation_float_tex, nx , ny, pitchf1*sizeof(float)) );
	gaussBlurSeparateMirrorY_const_tex<<<dimGrid,dimBlock>>>(out_g,nx,ny,pitchf1,radius);
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));

	if(selfallocmask) delete [] mask;
	if(selfalloctemp) cuda_free(temp_g);
}

__global__ void resampleAreaParallelSeparateX_tex
(
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float hx,
		float factor = 0.0f
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;

	if(factor == 0.0f) factor = 1.0f/hx;

	if(x<nx&&y<ny){
		float px = (float)x * hx;
		float left = ceil(px) - px;
		if(left > hx) left = hx;
		float midx = hx - left;
		float right = midx - floorf(midx);
		midx = midx - right;

		float out = 0.0f;
		if(left > 0.0f){
			out += tex2D(tex_linearoperation,px+LO_RS_AREA_OFFSET,y+LO_RS_AREA_OFFSET)*left*factor;
			px+= 1.0f;
		}
		while(midx > 0.0f){
			out += tex2D(tex_linearoperation,px+LO_RS_AREA_OFFSET,y+LO_RS_AREA_OFFSET)*factor;
			px += 1.0f;
			midx -= 1.0f;
		}
		if(right > RESAMPLE_EPSILON){
			out += tex2D(tex_linearoperation,px+LO_RS_AREA_OFFSET,y+LO_RS_AREA_OFFSET)*right*factor;
		}
		out_g[pg] = out;
	}
}

__global__ void resampleAreaParallelSeparateY_tex
(
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float hy,
		float factor = 0.0f
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;

	if(factor == 0.0f) factor = 1.0f/hy;

	if(x<nx&&y<ny){
		float py = (float)y * hy;
		float top = ceil(py) - py;
		if(top > hy) top = hy;
		float midy = hy - top;
		float bottom = midy - floorf(midy);
		midy = midy - bottom;

		float out = 0.0f;

		if(top > 0.0f){
			out += tex2D(tex_linearoperation,x+LO_RS_AREA_OFFSET,py+LO_RS_AREA_OFFSET)*top*factor;
			py += 1.0f;
		}
		while(midy > 0.0f){
			out += tex2D(tex_linearoperation,x+LO_RS_AREA_OFFSET,py+LO_RS_AREA_OFFSET)*factor;
			py += 1.0f;
			midy -= 1.0f;
		}
		if(bottom > RESAMPLE_EPSILON){
			out += tex2D(tex_linearoperation,x+LO_RS_AREA_OFFSET,py+LO_RS_AREA_OFFSET)*bottom*factor;
		}

		out_g[pg] = out;
	}
}





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
		float *help_g,
		float scalefactor
)
{
	bool selfalloc = help_g == NULL;
	if(selfalloc){
		fprintf(stderr,"\nADVICE: Use a helper array for separate Resampling!");
		cuda_malloc2D((void**)&help_g,nx_out,ny_in,1,sizeof(float),&pitchf1_out);
	}

	setTexturesLinearOperations(0);

	dim3 dimBlock = dim3(LO_BW,LO_BH);
	dim3 dimGrid = dim3((nx_out%LO_BW) ? (nx_out/LO_BW+1) : (nx_out/LO_BW),(ny_in%LO_BH) ? (ny_in/LO_BH+1) : (ny_in/LO_BH));
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, in_g,
		&linearoperation_float_tex, nx_in , ny_in, pitchf1_in*sizeof(float)) );
	resampleAreaParallelSeparateX_tex<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_in,pitchf1_out,(float)(nx_in)/(float)(nx_out),(float)(nx_out)/(float)(nx_in));
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));

	dimBlock = dim3(LO_BW,LO_BH);
	dimGrid = dim3((nx_out%LO_BW) ? (nx_out/LO_BW+1) : (nx_out/LO_BW),(ny_out%LO_BH) ? (ny_out/LO_BH+1) : (ny_out/LO_BH));
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, help_g,
		&linearoperation_float_tex, nx_out , ny_in, pitchf1_out*sizeof(float)) );
	resampleAreaParallelSeparateY_tex<<<dimGrid,dimBlock>>>
			(out_g,nx_out,ny_out,pitchf1_out,(float)(ny_in)/(float)(ny_out),(float)(ny_out)/(float)(ny_in)*scalefactor);
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));

	if(selfalloc){
		cuda_free(help_g);
	}
}

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
		float *help_g,
		float scalefactor
)
{
	bool selfalloc = help_g == NULL;
	if(selfalloc){
		fprintf(stderr,"\nADVICE: Use a helper array for separate Resampling!");
		cuda_malloc2D((void**)&help_g,nx_out,ny_in,1,sizeof(float),&pitchf1_out);
	}

	setTexturesLinearOperations(0);

	dim3 dimBlock = dim3(LO_BW,LO_BH);
	dim3 dimGrid = dim3((nx_out%LO_BW) ? (nx_out/LO_BW+1) : (nx_out/LO_BW),(ny_in%LO_BH) ? (ny_in/LO_BH+1) : (ny_in/LO_BH));
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, in_g,
		&linearoperation_float_tex, nx_in , ny_in, pitchf1_in*sizeof(float)) );
	resampleAreaParallelSeparateX_tex<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_in,pitchf1_out,(float)(nx_in)/(float)(nx_out),1.0f);
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));

	dimBlock = dim3(LO_BW,LO_BH);
	dimGrid = dim3((nx_out%LO_BW) ? (nx_out/LO_BW+1) : (nx_out/LO_BW),(ny_out%LO_BH) ? (ny_out/LO_BH+1) : (ny_out/LO_BH));
	cutilSafeCall( cudaBindTexture2D(0, &tex_linearoperation, help_g,
		&linearoperation_float_tex, nx_out , ny_in, pitchf1_out*sizeof(float)) );
	resampleAreaParallelSeparateY_tex<<<dimGrid,dimBlock>>>
			(out_g,nx_out,ny_out,pitchf1_out,(float)(ny_in)/(float)(ny_out),scalefactor);
	catchkernel;
	cutilSafeCall (cudaUnbindTexture(tex_linearoperation));

	if(selfalloc){
		cuda_free(help_g);
	}
}

__global__ void addKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	if(x<nx && y<ny){
		accumulator_g[pg] += increment_g[pg];
	}
}

__global__ void subKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	if(x<nx && y<ny){
		accumulator_g[pg] -= increment_g[pg];
	}
}

__global__ void setKernel
(
		float *field_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float value
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	if(x<nx && y<ny){
		field_g[pg] = value;
	}
}

