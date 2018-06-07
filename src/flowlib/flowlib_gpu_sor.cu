/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    flowlib_gpu_sor.cu
*
*
* implement all functions with ### implement me ### in the function body
\****************************************************************************/ 

/*
 * flowlib_gpu_sor.cu
 *
 *  Created on: Mar 14, 2012
 *      Author: steinbrf
 */

//#include <flowlib_gpu_sor.hpp>
#include "flowlib.hpp"
#include <auxiliary/cuda_basic.cuh>

#include <linearoperations/linearoperations.cuh>

#include <auxiliary/debug.hpp>

cudaChannelFormatDesc flow_sor_float_tex = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_flow_sor_I1;
texture<float, 2, cudaReadModeElementType> tex_flow_sor_I2;
bool textures_flow_sor_initialized = false;

#define IMAGE_FILTER_METHOD cudaFilterModeLinear
#define SF_TEXTURE_OFFSET 0.5f

#define SF_BW 16
#define SF_BH 16


FlowLibGpuSOR::FlowLibGpuSOR(int par_nx, int par_ny):
FlowLib(par_nx,par_ny),FlowLibGpu(par_nx,par_ny),FlowLibSOR(par_nx,par_ny)
{

	cuda_malloc2D((void**)&_penDat,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_penReg,_nx,_ny,1,sizeof(float),&_pitchf1);

	cuda_malloc2D((void**)&_b1,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_b2,_nx,_ny,1,sizeof(float),&_pitchf1);

}

FlowLibGpuSOR::~FlowLibGpuSOR()
{
	if(_penDat) cutilSafeCall(cudaFree(_penDat));
	if(_penReg) cutilSafeCall(cudaFree(_penReg));
	if(_b1)     cutilSafeCall(cudaFree(_b1));
	if(_b2)     cutilSafeCall(cudaFree(_b2));
}

void bind_textures(const float *I1_g, const float *I2_g, int nx, int ny, int pitchf1)
{
	tex_flow_sor_I1.addressMode[0] = cudaAddressModeClamp;
	tex_flow_sor_I1.addressMode[1] = cudaAddressModeClamp;
	tex_flow_sor_I1.filterMode = IMAGE_FILTER_METHOD ;
	tex_flow_sor_I1.normalized = false;

	tex_flow_sor_I2.addressMode[0] = cudaAddressModeClamp;
	tex_flow_sor_I2.addressMode[1] = cudaAddressModeClamp;
	tex_flow_sor_I2.filterMode = IMAGE_FILTER_METHOD;
	tex_flow_sor_I2.normalized = false;

	cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I1, I1_g,
		&flow_sor_float_tex, nx, ny, pitchf1*sizeof(float)) );
	cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I2, I2_g,
		&flow_sor_float_tex, nx, ny, pitchf1*sizeof(float)) );
}

void unbind_textures_flow_sor()
{
  cutilSafeCall (cudaUnbindTexture(tex_flow_sor_I1));
  cutilSafeCall (cudaUnbindTexture(tex_flow_sor_I2));
}

void update_textures_flow_sor(const float *I2_resampled_warped_g, int nx_fine, int ny_fine, int pitchf1)
{
	cutilSafeCall (cudaUnbindTexture(tex_flow_sor_I2));
	cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I2, I2_resampled_warped_g,
		&flow_sor_float_tex, nx_fine, ny_fine, pitchf1*sizeof(float)) );
}


/**
 * @brief Adds one flow field onto another
 * @param du_g Horizontal increment
 * @param dv_g Vertical increment
 * @param u_g Horizontal accumulation
 * @param v_g Vertical accumulation
 * @param nx Image width
 * @param ny Image height
 * @param pitchf1 Image pitch for single float images
 */
__global__ void add_flow_fields
(
	const float *du_g,
	const float *dv_g,
	float *u_g,
	float *v_g,
	int    nx,
	int    ny,
	int    pitchf1
)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < nx && y < ny){
  	u_g[y*pitchf1+x] = u_g[y*pitchf1+x] + du_g[y*pitchf1+x];
  	v_g[y*pitchf1+x] = v_g[y*pitchf1+x] + dv_g[y*pitchf1+x];
  }
}


/**
 * @brief Kernel to compute the penalty values for several
 * lagged-diffusivity iterations taking into account pixel sizes for warping.
 * Image derivatives are read from texture, flow derivatives from shared memory
 * @param u_g Pointer to global device memory for the horizontal
 * flow component of the accumulation flow field
 * @param v_g Pointer to global device memory for the vertical
 * flow component of the accumulation flow field
 * @param du_g Pointer to global device memory for the horizontal
 * flow component of the increment flow field
 * @param dv_g Pointer to global device memory for the vertical
 * flow component of the increment flow field
 * @param penaltyd_g Pointer to global device memory for data term penalty
 * @param penaltyr_g Pointer to global device memory for regularity term
 * penalty
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param data_epsilon Smoothing parameter for the TV Penalization of the data
 * term
 * @param diff_epsilon Smoothing parameter for the TV Penalization of the
 * regularity term
 * @param pitchf1 Image pitch for single float images
 */
__global__ void sorflow_update_robustifications_warp_tex_shared
(
	const float *u_g,
	const float *v_g,
	const float *du_g,
	const float *dv_g,
	float *penaltyd_g,
	float *penaltyr_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  data_epsilon,
	float  diff_epsilon,
	int    pitchf1
)
{
	const float hx_1 = 1.0f / (2.0f*hx);
	const float hy_1 = 1.0f / (2.0f*hy);


	const int x = blockIdx.x * SF_BW + threadIdx.x;
	const int y = blockIdx.y * SF_BW + threadIdx.y;

	__shared__ float u1[SF_BW][SF_BH];
	__shared__ float u2[SF_BW][SF_BH];
	__shared__ float du[SF_BW][SF_BH];
	__shared__ float dv[SF_BW][SF_BH];

	const float xx = (float)x + SF_TEXTURE_OFFSET;
	const float yy = (float)y + SF_TEXTURE_OFFSET;

	double dxu, dxv, dyu, dyv, dataterm;
	float3 is;

	if(x < nx && y < ny){
		is.x = 0.5f*(    tex2D(tex_flow_sor_I1,xx+1.0f,yy)
				            -tex2D(tex_flow_sor_I1,xx-1.0f,yy)
				            +tex2D(tex_flow_sor_I2,xx+1.0f,yy)
				            -tex2D(tex_flow_sor_I2,xx-1.0f,yy))*hx_1;
		is.y = 0.5f*(    tex2D(tex_flow_sor_I1,xx,yy+1.0f)
				            -tex2D(tex_flow_sor_I1,xx,yy-1.0f)
				            +tex2D(tex_flow_sor_I2,xx,yy+1.0f)
				            -tex2D(tex_flow_sor_I2,xx,yy-1.0f))*hy_1;
		is.z = (tex2D(tex_flow_sor_I2,xx,yy)-tex2D(tex_flow_sor_I1,xx,yy));

		u1[threadIdx.x][threadIdx.y] = u_g[y*pitchf1+x];
		u2[threadIdx.x][threadIdx.y] = v_g[y*pitchf1+x];
		du[threadIdx.x][threadIdx.y] = du_g[y*pitchf1+x];
		dv[threadIdx.x][threadIdx.y] = dv_g[y*pitchf1+x];
	}

	__syncthreads();

	if(x < nx && y < ny){
		dxu = ((x<nx-1 ? (threadIdx.x<SF_BW-1 ? u1[threadIdx.x+1][threadIdx.y] : u_g[y*pitchf1+x+1])   : u1[threadIdx.x][threadIdx.y])
				- (x>0     ? (threadIdx.x>0       ? u1[threadIdx.x-1][threadIdx.y] : u_g[y*pitchf1+x-1])   : u1[threadIdx.x][threadIdx.y]))*hx_1;
		dyu = ((y<ny-1 ? (threadIdx.y<SF_BH-1 ? u1[threadIdx.x][threadIdx.y+1] : u_g[(y+1)*pitchf1+x]) : u1[threadIdx.x][threadIdx.y])
				- (y>0     ? (threadIdx.y>0       ? u1[threadIdx.x][threadIdx.y-1] : u_g[(y-1)*pitchf1+x]) : u1[threadIdx.x][threadIdx.y]))*hy_1;
		dxv = ((x<nx-1 ? (threadIdx.x<SF_BW-1 ? u2[threadIdx.x+1][threadIdx.y] : v_g[y*pitchf1+x+1])   : u2[threadIdx.x][threadIdx.y])
				- (x>0     ? (threadIdx.x>0       ? u2[threadIdx.x-1][threadIdx.y] : v_g[y*pitchf1+x-1])   : u2[threadIdx.x][threadIdx.y]))*hx_1;
		dyv = ((y<ny-1 ? (threadIdx.y<SF_BH-1 ? u2[threadIdx.x][threadIdx.y+1] : v_g[(y+1)*pitchf1+x]) : u2[threadIdx.x][threadIdx.y])
				- (y>0     ? (threadIdx.y>0       ? u2[threadIdx.x][threadIdx.y-1] : v_g[(y-1)*pitchf1+x]) : u2[threadIdx.x][threadIdx.y]))*hy_1;

		dxu += ((x<nx-1 ? (threadIdx.x<SF_BW-1 ? du[threadIdx.x+1][threadIdx.y] : du_g[y*pitchf1+x+1])   : du[threadIdx.x][threadIdx.y])
				-  (x>0     ? (threadIdx.x>0       ? du[threadIdx.x-1][threadIdx.y] : du_g[y*pitchf1+x-1])   : du[threadIdx.x][threadIdx.y]))*hx_1;
		dyu += ((y<ny-1 ? (threadIdx.y<SF_BH-1 ? du[threadIdx.x][threadIdx.y+1] : du_g[(y+1)*pitchf1+x]) : du[threadIdx.x][threadIdx.y])
				-  (y>0     ? (threadIdx.y>0       ? du[threadIdx.x][threadIdx.y-1] : du_g[(y-1)*pitchf1+x]) : du[threadIdx.x][threadIdx.y]))*hy_1;
		dxv += ((x<nx-1 ? (threadIdx.x<SF_BW-1 ? dv[threadIdx.x+1][threadIdx.y] : dv_g[y*pitchf1+x+1])   : dv[threadIdx.x][threadIdx.y])
				-  (x>0     ? (threadIdx.x>0       ? dv[threadIdx.x-1][threadIdx.y] : dv_g[y*pitchf1+x-1])   : dv[threadIdx.x][threadIdx.y]))*hx_1;
		dyv += ((y<ny-1 ? (threadIdx.y<SF_BH-1 ? dv[threadIdx.x][threadIdx.y+1] : dv_g[(y+1)*pitchf1+x]) : dv[threadIdx.x][threadIdx.y])
				-  (y>0     ? (threadIdx.y>0       ? dv[threadIdx.x][threadIdx.y-1] : dv_g[(y-1)*pitchf1+x]) : dv[threadIdx.x][threadIdx.y]))*hy_1;

		dataterm = du[threadIdx.x][threadIdx.y]*is.x + dv[threadIdx.x][threadIdx.y]*is.y + is.z;

		penaltyd_g[y*pitchf1+x] = 1.0f / sqrt(dataterm*dataterm + data_epsilon);
		penaltyr_g[y*pitchf1+x] = 1.0f / sqrt(dxu*dxu + dxv*dxv + dyu*dyu + dyv*dyv + diff_epsilon);
	}
}


/**
 * @brief Precomputes one value as the sum of all values not depending of the
 * current flow increment
 * @param u_g Pointer to global device memory for the horizontal
 * flow component of the accumulation flow field
 * @param v_g Pointer to global device memory for the vertical
 * flow component of the accumulation flow field
 * @param penaltyd_g Pointer to global device memory for data term penalty
 * @param penaltyr_g Pointer to global device memory for regularity term
 * penalty
 * @param bu_g Pointer to global memory for horizontal result value
 * @param bv_g Pointer to global memory for vertical result value
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param pitchf1 Image pitch for single float images
 */
__global__ void sorflow_update_righthandside_shared
(
	const float *u_g,
	const float *v_g,
	const float *penaltyd_g,
	const float *penaltyr_g,
	float *bu_g,
	float *bv_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	int    pitchf1
)
{
	const float hx_1 = 1.0f / (2.0f*hx);
	const float hy_1 = 1.0f / (2.0f*hy);

	const float hx_2=lambda/(hx*hx);
	const float hy_2=lambda/(hy*hy);

	const int x = blockIdx.x * SF_BW + threadIdx.x;
	const int y = blockIdx.y * SF_BW + threadIdx.y;

	__shared__ float u1[SF_BW][SF_BH];
	__shared__ float u2[SF_BW][SF_BH];
	__shared__ float pend[SF_BW][SF_BH];
	__shared__ float penr[SF_BW][SF_BH];

	float IxIt, IyIt;

	const float xx = (float)x + SF_TEXTURE_OFFSET;
	const float yy = (float)y + SF_TEXTURE_OFFSET;

	float3 is;

	float xp, xm, yp, ym, sum;

	if(x < nx && y < ny){
		is.x = 0.5f*(tex2D(tex_flow_sor_I1,xx+1.0f,yy)
				            -tex2D(tex_flow_sor_I1,xx-1.0f,yy)
				            +tex2D(tex_flow_sor_I2,xx+1.0f,yy)
				            -tex2D(tex_flow_sor_I2,xx-1.0f,yy))*hx_1;
		is.y = 0.5f*(tex2D(tex_flow_sor_I1,xx,yy+1.0f)
				            -tex2D(tex_flow_sor_I1,xx,yy-1.0f)
				            +tex2D(tex_flow_sor_I2,xx,yy+1.0f)
				            -tex2D(tex_flow_sor_I2,xx,yy-1.0f))*hy_1;
		is.z = (tex2D(tex_flow_sor_I2,xx,yy)-tex2D(tex_flow_sor_I1,xx,yy));

		IxIt = is.x*is.z;
		IyIt = is.y*is.z;

		u1[threadIdx.x][threadIdx.y] = u_g[y*pitchf1+x];
		u2[threadIdx.x][threadIdx.y] = v_g[y*pitchf1+x];

		pend[threadIdx.x][threadIdx.y] = penaltyd_g[y*pitchf1+x];
		penr[threadIdx.x][threadIdx.y] = penaltyr_g[y*pitchf1+x];
	}

	__syncthreads();

	if(x < nx && y < ny){
		xp = (x<nx-1 ? ((threadIdx.x<SF_BW-1 ? penr[threadIdx.x+1][threadIdx.y] : penaltyr_g[y*pitchf1+x+1])  + penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hx_2;
		xm = (x>0    ? ((threadIdx.x>0       ? penr[threadIdx.x-1][threadIdx.y] : penaltyr_g[y*pitchf1+x-1])  + penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hx_2;
		yp = (y<ny-1 ? ((threadIdx.y<SF_BH-1 ? penr[threadIdx.x][threadIdx.y+1] : penaltyr_g[(y+1)*pitchf1+x])+ penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hy_2;
		ym = (y>0    ? ((threadIdx.y>0       ? penr[threadIdx.x][threadIdx.y-1] : penaltyr_g[(y-1)*pitchf1+x])+ penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hy_2;
		sum = xp + xm + yp + ym;


		bu_g[y*pitchf1+x] = - pend[threadIdx.x][threadIdx.y] * IxIt
		    + (x>0    ? xm*(threadIdx.x>0       ? u1[threadIdx.x-1][threadIdx.y] : u_g[y*pitchf1+x-1])   : 0.0f)
		    + (x<nx-1 ? xp*(threadIdx.x<SF_BW-1 ? u1[threadIdx.x+1][threadIdx.y] : u_g[y*pitchf1+x+1])   : 0.0f)
		    + (y>0    ? ym*(threadIdx.y>0       ? u1[threadIdx.x][threadIdx.y-1] : u_g[(y-1)*pitchf1+x]) : 0.0f)
		    + (y<ny-1 ? yp*(threadIdx.y<SF_BH-1 ? u1[threadIdx.x][threadIdx.y+1] : u_g[(y+1)*pitchf1+x]) : 0.0f)
				- sum * u1[threadIdx.x][threadIdx.y];

		bv_g[y*pitchf1+x] = - pend[threadIdx.x][threadIdx.y] * IyIt
	       +(x>0    ? xm*(threadIdx.x>0       ? u2[threadIdx.x-1][threadIdx.y] : v_g[y*pitchf1+x-1])   : 0.0f)
	      + (x<nx-1 ? xp*(threadIdx.x<SF_BW-1 ? u2[threadIdx.x+1][threadIdx.y] : v_g[y*pitchf1+x+1])   : 0.0f)
	      + (y>0    ? ym*(threadIdx.y>0       ? u2[threadIdx.x][threadIdx.y-1] : v_g[(y-1)*pitchf1+x]) : 0.0f)
	      + (y<ny-1 ? yp*(threadIdx.y<SF_BH-1 ? u2[threadIdx.x][threadIdx.y+1] : v_g[(y+1)*pitchf1+x]) : 0.0f)
				- sum * u2[threadIdx.x][threadIdx.y];
	}
}


/**
 * @brief Kernel to compute one Red-Black-SOR iteration for the nonlinear
 * Euler-Lagrange equation taking into account penalty values and pixel
 * size for warping
 * @param bu_g Right-Hand-Side values for horizontal flow
 * @param bv_g Right-Hand-Side values for vertical flow
 * @param penaltyd_g Pointer to global device memory holding data term penalization
 * @param penaltyr_g Pointer to global device memory holding regularity term
 * penalization
 * @param du_g Pointer to global device memory for the horizontal
 * flow component increment
 * @param dv_g Pointer to global device memory for the vertical
 * flow component increment
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param relaxation Overrelaxation for the SOR-solver
 * @param red Parameter deciding whether the red or black fields of a
 * checkerboard pattern are being updated
 * @param pitchf1 Image pitch for single float images
 */
__global__ void sorflow_nonlinear_warp_sor_shared
(
	const float *bu_g,
	const float *bv_g,
	const float *penaltyd_g,
	const float *penaltyr_g,
	float *du_g,
	float *dv_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	float  relaxation,
	int    red,
	int    pitchf1
)
{
	const float hx_1 = 1.0f / (2.0f*hx);
	const float hy_1 = 1.0f / (2.0f*hy);

	const float hx_2=lambda/(hx*hx);
	const float hy_2=lambda/(hy*hy);

	const int x = blockIdx.x * SF_BW + threadIdx.x;
	const int y = blockIdx.y * SF_BW + threadIdx.y;

	__shared__ float du[SF_BW][SF_BH];
	__shared__ float dv[SF_BW][SF_BH];
	__shared__ float pend[SF_BW][SF_BH];
	__shared__ float penr[SF_BW][SF_BH];

	const float xx = (float)x + SF_TEXTURE_OFFSET;
	const float yy = (float)y + SF_TEXTURE_OFFSET;

	float IxIx, IxIy, IyIy;

	float bu, bv;
	float3 is;


	float xp, xm, yp, ym, sum;

	if(x < nx && y < ny){
		is.x = 0.5f*(tex2D(tex_flow_sor_I1,xx+1.0f,yy)
				            -tex2D(tex_flow_sor_I1,xx-1.0f,yy)
				            +tex2D(tex_flow_sor_I2,xx+1.0f,yy)
				            -tex2D(tex_flow_sor_I2,xx-1.0f,yy))*hx_1;
		is.y = 0.5f*(tex2D(tex_flow_sor_I1,xx,yy+1.0f)
				            -tex2D(tex_flow_sor_I1,xx,yy-1.0f)
				            +tex2D(tex_flow_sor_I2,xx,yy+1.0f)
				            -tex2D(tex_flow_sor_I2,xx,yy-1.0f))*hy_1;
		is.z = (tex2D(tex_flow_sor_I2,xx,yy)-tex2D(tex_flow_sor_I1,xx,yy));

		IxIx = is.x*is.x;
		IxIy = is.x*is.y;
		IyIy = is.y*is.y;


		bu = bu_g[y*pitchf1+x];
		bv = bv_g[y*pitchf1+x];

		du[threadIdx.x][threadIdx.y] = du_g[y*pitchf1+x];
		dv[threadIdx.x][threadIdx.y] = dv_g[y*pitchf1+x];

		pend[threadIdx.x][threadIdx.y] = penaltyd_g[y*pitchf1+x];
		penr[threadIdx.x][threadIdx.y] = penaltyr_g[y*pitchf1+x];
	}

	__syncthreads();


	if(x < nx && y < ny && ((x+y)%2) == red){
		xp = (x<nx-1 ? ((threadIdx.x<SF_BW-1 ? penr[threadIdx.x+1][threadIdx.y] : penaltyr_g[y*pitchf1+x+1])  + penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hx_2;
		xm = (x>0    ? ((threadIdx.x>0 ?       penr[threadIdx.x-1][threadIdx.y] : penaltyr_g[y*pitchf1+x-1])  + penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hx_2;
		yp = (y<ny-1 ? ((threadIdx.y<SF_BH-1 ? penr[threadIdx.x][threadIdx.y+1] : penaltyr_g[(y+1)*pitchf1+x])+ penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hy_2;
		ym = (y>0    ? ((threadIdx.y>0 ?       penr[threadIdx.x][threadIdx.y-1] : penaltyr_g[(y-1)*pitchf1+x])+ penr[threadIdx.x][threadIdx.y])*0.5f : 0.0f)*hy_2;
		sum = xp + xm + yp + ym;


		du_g[y*pitchf1+x] = (1.0f-relaxation)*du[threadIdx.x][threadIdx.y]
		                  + relaxation * ( bu - pend[threadIdx.x][threadIdx.y]*IxIy*dv[threadIdx.x][threadIdx.y]
				    + (x>0    ? xm*(threadIdx.x>0       ? du[threadIdx.x-1][threadIdx.y] : du_g[y*pitchf1+x-1])   : 0.0f)
				    + (x<nx-1 ? xp*(threadIdx.x<SF_BW-1 ? du[threadIdx.x+1][threadIdx.y] : du_g[y*pitchf1+x+1])   : 0.0f)
				    + (y>0    ? ym*(threadIdx.y>0       ? du[threadIdx.x][threadIdx.y-1] : du_g[(y-1)*pitchf1+x]) : 0.0f)
				    + (y<ny-1 ? yp*(threadIdx.y<SF_BH-1 ? du[threadIdx.x][threadIdx.y+1] : du_g[(y+1)*pitchf1+x]) : 0.0f)
						) / (pend[threadIdx.x][threadIdx.y]*IxIx + sum);


		dv_g[y*pitchf1+x] = (1.0f-relaxation)*dv[threadIdx.x][threadIdx.y]
		                  + relaxation * (bv - pend[threadIdx.x][threadIdx.y]*IxIy*du[threadIdx.x][threadIdx.y]
			      + (x>0    ? xm*(threadIdx.x>0       ? dv[threadIdx.x-1][threadIdx.y] : dv_g[y*pitchf1+x-1])   : 0.0f)
			      + (x<nx-1 ? xp*(threadIdx.x<SF_BW-1 ? dv[threadIdx.x+1][threadIdx.y] : dv_g[y*pitchf1+x+1])   : 0.0f)
			      + (y>0    ? ym*(threadIdx.y>0       ? dv[threadIdx.x][threadIdx.y-1] : dv_g[(y-1)*pitchf1+x]) : 0.0f)
			      + (y<ny-1 ? yp*(threadIdx.y<SF_BH-1 ? dv[threadIdx.x][threadIdx.y+1] : dv_g[(y+1)*pitchf1+x]) : 0.0f)
						)/ (pend[threadIdx.x][threadIdx.y]*IyIy + sum);
	}
}

/**
 * @brief Method that calls the sorflow_nonlinear_warp_sor_shared in a loop,
 * with an outer loop for computing the diffisivity values for
 * one level of a coarse-to-fine implementation.
 * @param u_g Pointer to global device memory for the horizontal
 * flow component
 * @param v_g Pointer to global device memory for the vertical
 * flow component
 * @param du_g Pointer to global device memory for the horizontal
 * flow component increment
 * @param dv_g Pointer to global device memory for the vertical
 * flow component increment
 * @param bu_g Right-Hand-Side values for horizontal flow
 * @param bv_g Right-Hand-Side values for vertical flow
 * @param penaltyd_g Pointer to global device memory holding data term penalization
 * @param penaltyr_g Pointer to global device memory holding regularity term
 * penalization
 * @param nx Image width
 * @param ny Image height
 * @param pitchf1 Image pitch for single float images
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param outer_iterations Number of iterations of the penalty computation
 * @param inner_iterations Number of iterations for the SOR-solver
 * @param relaxation Overrelaxation for the SOR-solver
 * @param data_epsilon Smoothing parameter for the TV Penalization of the data
 * term
 * @param diff_epsilon Smoothing parameter for the TV Penalization of the
 * regularity term
 */
void sorflow_gpu_nonlinear_warp_level
(
		const float *u_g,
		const float *v_g,
		float *du_g,
		float *dv_g,
		float *bu_g,
		float *bv_g,
		float *penaltyd_g,
		float *penaltyr_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float hx,
		float hy,
		float lambda,
		float overrelaxation,
		int   outer_iterations,
		int   inner_iterations,
		float data_epsilon,
		float diff_epsilon
)
{
	int i, j;

	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BW) ? ((ny/SF_BW)+1) : (ny/SF_BW);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);

	cutilSafeCall( cudaMemset(du_g,0,pitchf1*ny*sizeof(float)));
	cutilSafeCall( cudaMemset(dv_g,0,pitchf1*ny*sizeof(float)));

	for(i=0;i<outer_iterations;i++){
		sorflow_update_robustifications_warp_tex_shared<<<dimGrid,dimBlock>>>
				(u_g,v_g,du_g,dv_g,penaltyd_g,penaltyr_g,nx,ny,hx,hy,
						data_epsilon,diff_epsilon,pitchf1);
		catchkernel;

		sorflow_update_righthandside_shared<<<dimGrid,dimBlock>>>
				(u_g,v_g,penaltyd_g,penaltyr_g,bu_g,bv_g,nx,ny,hx,hy,lambda,pitchf1);
		catchkernel;

		for(j=0;j<inner_iterations;j++){
			sorflow_nonlinear_warp_sor_shared<<<dimGrid,dimBlock>>>
					(bu_g,bv_g,penaltyd_g,penaltyr_g,du_g,dv_g,nx,ny,hx,hy,lambda,overrelaxation,0,pitchf1);
			catchkernel;

			sorflow_nonlinear_warp_sor_shared<<<dimGrid,dimBlock>>>
					(bu_g,bv_g,penaltyd_g,penaltyr_g,du_g,dv_g,nx,ny,hx,hy,lambda,overrelaxation,1,pitchf1);
			catchkernel;
		}
	}
}


float FlowLibGpuSOR::computeFlow()
{
	float lambda = _lambda * 255.0f;

	int   max_rec_depth;
	int   warp_max_levels;
	int   rec_depth;

	int ngx = (_nx%SF_BW) ? ((_nx/SF_BW)+1) : (_nx/SF_BW);
	int ngy = (_ny%SF_BH) ? ((_ny/SF_BH)+1) : (_ny/SF_BH);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);

	warp_max_levels = computeMaxWarpLevels();

  max_rec_depth = (((_start_level+1) < warp_max_levels) ?
                    (_start_level+1) : warp_max_levels) -1;

  if(max_rec_depth >= _I1pyramid->nl){
  	max_rec_depth = _I1pyramid->nl-1;
  }


  if(!textures_flow_sor_initialized){
		tex_flow_sor_I1.addressMode[0] = cudaAddressModeClamp;
		tex_flow_sor_I1.addressMode[1] = cudaAddressModeClamp;
		tex_flow_sor_I1.filterMode = IMAGE_FILTER_METHOD;
		tex_flow_sor_I1.normalized = false;

		tex_flow_sor_I2.addressMode[0] = cudaAddressModeClamp;
		tex_flow_sor_I2.addressMode[1] = cudaAddressModeClamp;
		tex_flow_sor_I2.filterMode = IMAGE_FILTER_METHOD;
		tex_flow_sor_I2.normalized = false;
		textures_flow_sor_initialized = true;
  }

  int nx_fine, ny_fine, nx_coarse=0, ny_coarse=0;

	float hx_fine;
	float hy_fine;

	cutilSafeCall( cudaMemset(_u1_g,0,_pitchf1*_ny*sizeof(float)));
	cutilSafeCall( cudaMemset(_u2_g,0,_pitchf1*_ny*sizeof(float)));

	for(rec_depth = max_rec_depth; rec_depth >= 0; rec_depth--){
	    if(_verbose) fprintf(stderr," Level %i",rec_depth);
		nx_fine = _I1pyramid->nx[rec_depth];
		ny_fine = _I1pyramid->ny[rec_depth];

		hx_fine=(float)_nx/(float)nx_fine;
		hy_fine=(float)_ny/(float)ny_fine;

		cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I1, _I1pyramid->level[rec_depth],
																				&flow_sor_float_tex, nx_fine, ny_fine,
																				_I1pyramid->pitch[rec_depth]*sizeof(float)) );

		if(_debug){
			sprintf(_debugbuffer,"debug/GI1 %i.png",rec_depth);
			saveCudaImage(_debugbuffer,_I1pyramid->level[rec_depth],nx_fine,ny_fine,_I1pyramid->pitch[rec_depth],1,1.0f,-1.0f);
			sprintf(_debugbuffer,"debug/GI2 %i.png",rec_depth);
			saveCudaImage(_debugbuffer,_I2pyramid->level[rec_depth],nx_fine,ny_fine,_I2pyramid->pitch[rec_depth],1,1.0f,-1.0f);
		}

		if(rec_depth < max_rec_depth){
			resampleAreaParallelSeparate(_u1_g,_u1_g,nx_coarse,ny_coarse,_pitchf1,nx_fine,ny_fine,_pitchf1,_b1);
			resampleAreaParallelSeparate(_u2_g,_u2_g,nx_coarse,ny_coarse,_pitchf1,nx_fine,ny_fine,_pitchf1,_b2);
		}

		if(rec_depth >= _end_level){
			backwardRegistrationBilinearFunctionTex(_I2pyramid->level[rec_depth],_u1_g,_u2_g,
			    _I2warp,_I1pyramid->level[rec_depth],
					nx_fine,ny_fine,_I2pyramid->pitch[rec_depth],_pitchf1,hx_fine,hy_fine);

			if(_debug){
				sprintf(_debugbuffer,"debug/GW2 %i.png",rec_depth);
				saveCudaImage(_debugbuffer,_I2warp,nx_fine,ny_fine,_pitchf1,1,1.0f,-1.0f);
			}

			cutilSafeCall (cudaUnbindTexture(tex_flow_sor_I2));

			cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I2, _I2warp,
																					&flow_sor_float_tex, nx_fine, ny_fine,
																					_pitchf1*sizeof(float)) );

			sorflow_gpu_nonlinear_warp_level
			(_u1_g,_u2_g,_u1lvl,_u2lvl,_b1,_b2,_penDat,_penReg,
					nx_fine,ny_fine,_pitchf1,
					hx_fine,hy_fine,
					lambda,_overrelaxation,
					_oi,_ii,
					_dat_epsilon,_reg_epsilon);

			add_flow_fields<<<dimGrid,dimBlock>>>(_u1lvl,_u2lvl,_u1_g,_u2_g,nx_fine,ny_fine,_pitchf1);
			catchkernel;
		}
		else{
			if(_verbose) fprintf(stderr," skipped");
		}

		nx_coarse = nx_fine;
		ny_coarse = ny_fine;
	}

	if(_debug) delete [] _debugbuffer;

  unbind_textures_flow_sor();

  //TODO: Timer
  return -1.0f;
}

