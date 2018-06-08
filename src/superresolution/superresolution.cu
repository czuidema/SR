/*
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    superresolution.cu
*
*
* implement all functions with ### implement me ### in the function body
\****************************************************************************/ 

/*
 * superresolution.cu
 *
 *  Created on: May 16, 2012
 *      Author: steinbrf
 */
#include "superresolution.cuh"
#include <stdio.h>
//#include <cutil.h>
//#include <cutil_inline.h>
#include <auxiliary/cuda_basic.cuh>
#include <vector>
#include <list>

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <filesystem/filesystem.h>


//#include <linearoperations.cuh>
#include <linearoperations/linearoperations.cuh>


#include "superresolution_definitions.h"

#include <auxiliary/debug.hpp>
#include "cusparse.h"
#include <cuda_runtime.h>

#ifdef DGT400
#define SR_BW 32
#define SR_BH 16
#else
#define SR_BW 16
#define SR_BH 16
#endif

#include <linearoperations/linearoperations.h>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



extern __shared__ float smem[];


__global__ void dualTVHuber_kernel
(
		const float *u_overrelaxed_g,
		float       *xi1_g,
		float       *xi2_g,
		int         nx,
		int         ny,
		int         pitchf1,
		float       factor_tv_update,
		float       factor_tv_clipping,
		float       huber_denom_tv,
		float       tau_dual
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	const int ps = threadIdx.y*blockDim.x+threadIdx.x;

	float *u_s = smem;

	if(x<nx && y<ny){
		u_s[ps] = u_overrelaxed_g[pg];
	}
	__syncthreads();

	if(x<nx && y<ny){
		float dx = xi1_g[pg] + tau_dual * factor_tv_update *
				(((x==nx-1) ? u_s[ps] : ((threadIdx.x==blockDim.x-1) ? u_overrelaxed_g[pg+1] : u_s[ps+1])) - u_s[ps])/huber_denom_tv;
		float dy = xi2_g[pg] + tau_dual * factor_tv_update *
				(((y==ny-1) ? u_s[ps] : ((threadIdx.y==blockDim.y-1) ? u_overrelaxed_g[pg+pitchf1] : u_s[ps+blockDim.x])) - u_s[ps])/huber_denom_tv;
		float denom = sqrtf(dx*dx + dy*dy)/factor_tv_clipping;
		if(denom < 1.0f) denom = 1.0f;
		xi1_g[pg] = dx / denom;
		xi2_g[pg] = dy / denom;
	}
}


__global__ void dualL1Difference_kernel
(
		const float *primal_g,
		const float *constant_g,
		float       *dual_g,
		int         nx,
		int         ny,
		int         pitchf1,
		float       factor_update,
		float       factor_clipping,
		float       huber_denom,
		float       tau_d
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	if(x<nx&&y<ny){
		float q = (dual_g[pg] + tau_d * factor_update * (primal_g[pg] - constant_g[pg]))/huber_denom;
		if(q < -factor_clipping) q = -factor_clipping;
		if(q >  factor_clipping) q =  factor_clipping;
		dual_g[pg] = q;
	}
}


__global__ void primal1N_kernel
(
		const float *dataterm_g,
		const float *p1_g,
		const float *p2_g,
		float *u_g,
		float *u_overrelaxed_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float factor_tv_update,
		float factor_degrade_update,
		float tau_p,
		float overrelaxation
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	const int pg = y*pitchf1+x;
	const int ps = threadIdx.y*blockDim.x+threadIdx.x;

	float *p1_s = smem;
	float *p2_s = p1_s + blockDim.x*blockDim.y;

	if(x<nx && y<ny){
		p1_s[ps] = p1_g[pg];
		p2_s[ps] = p2_g[pg];
	}
	__syncthreads();

	if(x<nx && y<ny){
		float div = p1_s[ps] - ((x==0) ? 0.0f : ((threadIdx.x==0) ? p1_g[pg-1] : p1_s[ps-1])) +
				        p2_s[ps] - ((y==0) ? 0.0f : ((threadIdx.y==0) ? p2_g[pg-pitchf1] : p2_s[ps-blockDim.x]));
		float u_old = u_g[pg];
		float u_new = u_old + tau_p * (factor_tv_update * div - factor_degrade_update * dataterm_g[pg]);
		u_g[pg] = u_new;
		u_overrelaxed_g[pg] = overrelaxation * u_new + (1.0f-overrelaxation) *  u_old;
	}
}


void getData( std::string filename, std::string delimeter, std::vector <int> & rowIndex, std::vector <int> & columnIndex, std::vector <float> & values )
{
        std::ifstream file(filename.c_str());
        std::string line = "";
        int lineNumber = 0;
        // Iterate through each line and split the content using delimeter
        while (getline(file, line))
        {
                if(lineNumber < 3){
                        ++lineNumber;
                        std::vector<std::string> vec;
                        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));

                        switch(lineNumber){
                                case 1:
                                        for(int i = 0; i < vec.size(); ++i){
                                                int num = atoi(vec.at(i).c_str());
                                                rowIndex.push_back(num);
                                        }
                                        break;
                                case 2:
                                        for(int i = 0; i < vec.size(); ++i){
                                                int num = atoi(vec.at(i).c_str());
                                                columnIndex.push_back(num);
                                        }
                                        break;
                                case 3:
                                        for(int i = 0; i < vec.size(); ++i){
                                                int num = atoi(vec.at(i).c_str());
                                                values.push_back(num);
                                        }
                                        break;
                        }
                }
                else{
                }
        }

        // Close the File
        file.close();
}



void computeSuperresolutionUngerGPU
(
		float *xi1_g,
		float *xi2_g,
		float *atlasLowRes,
		float *temp2_g,
		float *atlasHighRes,
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
)
{

	fprintf(stderr,"\nComputing 1N Superresolution from %i Images on GPU",(int)images_g.size());
	
	
	// initialization of matrix multiplication help variables.
	cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
	cusparseStatus_t status;
    	cusparseHandle_t handle=0;
	cusparseMatDescr_t descr=0;
    	int *    cooRowIndexHostPtr=0;
    	int *    cooColIndexHostPtr=0;
    	float * cooValHostPtr=0;
    	int *    cooRowIndex=0;
    	int *    cooColIndex=0;
    	float * cooVal=0;
    	float * yHostPtr=0;
    	float * y=0;
    	int *    csrRowPtr=0;
    	int      n, nnz;
    	float alpha = 1.0;
    	float beta = 0.0;
	std::string matricesFolderPath = "data/projectionMatricesSR";
    	std::vector <int> rowIndexVector, columnIndexVector;
    	std::vector <float>  valuesVector;
    	
	status = cusparseCreate(&handle);
        status= cusparseCreateMatDescr(&descr);

        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);



	// load all projection matrices.
	Folder folder(matricesFolderPath);
	std::vector<std::string> matrixNames = folder.getFilesWithEndingVector(".csv");
	std::string path = "";
	int numOfFiles = matrixNames.size();
	fprintf(stderr,"\nNumber of files: %d",numOfFiles);
	std::vector<int> rowIndexAllVector, columnIndexAllVector, nnzAllVector;
	std::vector<float> valuesAllVector;
	for(int i = 0; i < numOfFiles; ++i){
		fprintf(stderr,"\nIn the files loading loop: %i", i);
		path = matricesFolderPath + matrixNames[i]; 
		getData(path, ",", rowIndexVector, columnIndexVector, valuesVector);

		rowIndexAllVector.insert(rowIndexAllVector.end(), rowIndexVector.begin(), rowIndexVector.end());
		columnIndexAllVector.insert(columnIndexAllVector.end(), columnIndexVector.begin(), columnIndexVector.end());
		valuesAllVector.insert(valuesAllVector.end(), valuesVector.begin(), valuesVector.end());
		nnzAllVector.push_back(valuesVector.size());
		path = "";
	}

	int numberOfMatrices = nnzAllVector.size();
        n = nx*ny; // image dimensions
        int sumNNZ = 0;

	fprintf(stderr,"\nNumber of matrices: %i",numberOfMatrices);
		
	// projected image
	float* projectedImage;
	gpuErrchk( cudaMalloc((void**)&projectedImage, n*sizeof(projectedImage[0])) );

    	dim3 dimBlock(SR_BW,SR_BH);
	dim3 dimGrid((nx%SR_BW) ? (nx/SR_BW+1) : (nx/SR_BW),(ny%SR_BH) ? (ny/SR_BH+1) : (ny/SR_BH));
	dim3 dimGridOrig((nx_orig%SR_BW) ? (nx_orig/SR_BW+1) : (nx_orig/SR_BW),(ny_orig%SR_BH) ? (ny_orig/SR_BH+1) : (ny_orig/SR_BH));

	cutilSafeCall( cudaMemset(xi1_g,0,pitchf1*ny*sizeof(float)));
	cutilSafeCall( cudaMemset(xi2_g,0,pitchf1*ny*sizeof(float)));
	for(unsigned int k=0;k<q_g.size();k++){
		cutilSafeCall( cudaMemset(q_g[k],0,pitchf1_orig*ny_orig*sizeof(float)));
	}
	setKernel<<<dimGrid,dimBlock>>>(u_g,nx,ny,pitchf1,64.0f);
	setKernel<<<dimGrid,dimBlock>>>(uor_g,nx,ny,pitchf1,64.0f);
	catchkernel;

	float factor_tv_update = pow(factor_tv,CLIPPING_TRADEOFF_TV);
	float factor_tv_clipping = factor_tv/factor_tv_update;
	float huber_denom_tv = 1.0f + huber_epsilon*tau_d/factor_tv;

	float factorquad = factor_rescale_x*factor_rescale_y*factor_rescale_x*factor_rescale_y;
	float factor_degrade_update = pow(factorquad,CLIPPING_TRADEOFF_DEGRADE_1N);
	float factor_degrade_clipping = factorquad/factor_degrade_update;
	float huber_denom_degrade = 1.0f + huber_epsilon*tau_d/factor_degrade_clipping;



	for(int i=0;i<oi;i++){
		fprintf(stderr," %i",i);
		//DUAL TV
		dualTVHuber_kernel<<<dimGrid,dimBlock,dimBlock.x*dimBlock.y*sizeof(float)>>>
				(uor_g,xi1_g,xi2_g,nx,ny,pitchf1,factor_tv_update,factor_tv_clipping,huber_denom_tv,tau_d);
		catchkernel;



		//DUAl DATA
		float tau_data = tau_d;
		unsigned int k=0;
		std::vector<float*>::iterator image = images_g.begin();
                std::list<FlowGPU>::iterator flow = flowsGPU.begin();
		unsigned int k_m = 0;
		

		while( image != images_g.end() && flow != flowsGPU.end() && k < q_g.size()  ){
			float *f_g = *image;
			backwardRegistrationBilinearValueTex(uor_g,flow->u_g,flow->v_g,projectedImage,0.0f,nx,ny,pitchf1,pitchf1,1.0f,1.0f);
			// PREPARING THE PROJECTION MATRICES.
                        
			if(k != 0)
			{ 

                                rowIndexVector.insert(rowIndexVector.begin(), columnIndexAllVector.begin() + sumNNZ,
                                                columnIndexAllVector.begin() + sumNNZ + nnzAllVector[k_m]);
			
                                columnIndexVector.insert(columnIndexVector.begin(), columnIndexAllVector.begin() + sumNNZ,
                                                columnIndexAllVector.begin() + sumNNZ + nnzAllVector[k_m]);
				
                                valuesVector.insert(valuesVector.begin(), valuesAllVector.begin() + sumNNZ,
                                                valuesAllVector.begin() + sumNNZ + nnzAllVector[k_m]);
				
			
                                sumNNZ += nnzAllVector[k_m];
                                cooRowIndexHostPtr = (int *)   malloc(nnz*sizeof(cooRowIndexHostPtr[0]));
                                cooColIndexHostPtr = (int *)   malloc(nnz*sizeof(cooColIndexHostPtr[0]));
                                cooValHostPtr = (float *)malloc(nnz*sizeof(cooValHostPtr[0]));
                                cooRowIndexHostPtr = &rowIndexVector[0];
                                cooColIndexHostPtr = &columnIndexVector[0];
                                cooValHostPtr = &valuesVector[0];
			        nnz=columnIndexVector.size();
                                cudaMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0]));
				catchkernel;
                                cudaStat2 = cudaMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0]));
                                catchkernel;
				cudaStat3 = cudaMalloc((void**)&cooVal,nnz*sizeof(cooVal[0]));
                                catchkernel;
				cudaStat4 = cudaMalloc((void**)&y,n*sizeof(y[0]));
				cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,
                                   (size_t)(nnz*sizeof(cooRowIndex[0])),
                                   cudaMemcpyHostToDevice);
				catchkernel;
                                cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr,
                                   (size_t)(nnz*sizeof(cooColIndex[0])),
                                   cudaMemcpyHostToDevice);
                                cudaStat3 = cudaMemcpy(cooVal,      cooValHostPtr,
                                   (size_t)(nnz*sizeof(cooVal[0])),
                                   cudaMemcpyHostToDevice);
				fprintf(stderr,"\nTEST 3");
                                gpuErrchk( cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0])) );
                                status = cusparseXcoo2csr(handle,cooRowIndex,nnz,n,
                                        csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
					
				catchkernel;
                                status= cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                                   &alpha, descr, cooVal, csrRowPtr, cooColIndex,
                                   &atlasLowRes[0], &beta, &projectedImage[0]);
							
                                
				k_m++;
                        }
                        // PREPARING PROJECTION MATRICES DONE.
			
			if(blur > 0.0f){
				gaussBlurSeparateMirrorGpu(projectedImage,temp2_g,nx,ny,pitchf1,blur,blur,(int)(3.0f*blur),temp4_g,NULL);
			}
			else{
				float *temp = projectedImage; projectedImage = temp2_g; temp2_g = temp;
			}

		while( image != images_g.end() && flow != flowsGPU.end() && k < q_g.size() ){
			resampleAreaParallelSeparateAdjoined(q_g[k],projectedImage,nx_orig,ny_orig,pitchf1_orig,nx,ny,pitchf1,temp4_g);
			if(blur > 0.0f){
				gaussBlurSeparateMirrorGpu(projectedImage,temp2_g,nx,ny,pitchf1,blur,blur,(int)(3.0f*blur),temp4_g,NULL);
			}
			else{
				float *temp = projectedImage; projectedImage = temp2_g; temp2_g = temp;
			}
			cutilSafeCall( cudaMemset(projectedImage,0,pitchf1*ny*sizeof(float)));
			forewardRegistrationBilinearAtomic(flow->u_g,flow->v_g,temp2_g,projectedImage,nx,ny,pitchf1);
			
                       // for cuSparse API input/output data has to be on the GPU.      
			
			addKernel<<<dimGrid,dimBlock>>>(projectedImage,atlasHighRes,nx,ny,pitchf1);
			catchkernel;
			k++;
			flow++;
			image++;
		}
		primal1N_kernel<<<dimGrid,dimBlock,dimBlock.x*dimBlock.y*sizeof(float)*2>>>
				(atlasHighRes,xi1_g,xi2_g,u_g,uor_g,nx,ny,pitchf1,factor_tv_update,factor_degrade_update,tau_p,overrelaxation);
		
		}
		atlasHighRes = projectedImage;	
		catchkernel;

	}
}





