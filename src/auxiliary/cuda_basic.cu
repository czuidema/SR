/***************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    cuda_basic.cu
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "cuda_basic.cuh"
#include <string>


//#define CUDABASICVERBOSE

#ifndef HALF_WARP_SIZE
#define HALF_WARP_SIZE 16
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifdef GT400
#define PADD_SIZE WARP_SIZE
#else
#define PADD_SIZE HALF_WARP_SIZE
#endif

//#define CUDABASICVERBOSE

int whole = 0;

bool cuda_malloc2D(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int *pitch)
{
	//cuInit(0);
    size_t pitchbytes;

	int manualpitch = (int)(nx*nc*type_size);
	while(manualpitch%PADD_SIZE)
	{
		manualpitch += (int)(nc*type_size);
	}
	manualpitch /= (int)(nc*type_size);

#ifdef CUDABASICVERBOSE
	fprintf(stderr,"\nTrying to Allocate:  %d -> %d ",
			nx*nc*type_size * ny,whole + nx*nc*type_size * ny);
#endif

	int pitchrequest = *pitch;

	cutilSafeCall( cudaMallocPitch(
			(void**) device_image, &pitchbytes, nx*nc*type_size, ny));
	*pitch = (int)(pitchbytes/(nc*type_size));

	//Padd Manually
	if((*pitch)*type_size*nc < pitchbytes)
	{
#ifdef CUDABASICVERBOSE
		fprintf(stderr,"\nERROR: Pitch Bytes are not a Multiple of Type Size:");
		fprintf(stderr,"\nWidth: %d | Channels: %d | TypeSize: %d | Pitchbytes: %d | Pitch: %d",
					 nx,nc,type_size,pitchbytes,*pitch);
#endif
		cutilSafeCall(cudaFree(*device_image));
		*pitch = manualpitch;
		cutilSafeCall(cudaMalloc(device_image,(*pitch)*nc*type_size*ny));
		pitchbytes = (*pitch)*nc*type_size;
#ifdef CUDABASICVERBOSE
		fprintf(stderr,"\nPadded the Image to Pitch %d",*pitch);
#endif
	}
	whole += (int)(pitchbytes * ny);
	if(pitchrequest > 0)
	{
		if(pitchrequest != (*pitch))
		{
			fprintf(stderr,"\n\nWARNING: Prior Pitch of %i"
					" is different to Pitch of %i\n\n",pitchrequest,*pitch);
		}
	}

#ifdef CUDABASICVERBOSE

	fprintf(stderr,"\t\tAllocated:  %d -> %d   ",
			pitchbytes * ny,whole);



	if(manualpitch != (*pitch))
	{
		fprintf(stderr,"\nWARNING: The pitch of %i (in bytes %i) "
				"is different from the manual pitch of %i, in bytes %i * %i = %i",
				*pitch,(int)pitchbytes,manualpitch,manualpitch,nc*type_size,manualpitch*nc*type_size);
	}
#endif
	return true;
}

bool cuda_malloc2D_manual(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int pitch)
{
	size_t pitchbytes;

	int manualpitch = (int)(nx*nc*type_size);
	while(manualpitch%PADD_SIZE)
	{
		manualpitch += (int)(nc*type_size);
	}
	manualpitch /= (int)(nc*type_size);

#ifdef CUDABASICVERBOSE
	fprintf(stderr,"\nTrying to Allocate:  %d -> %d ",
			nx*nc*type_size * ny,whole + nx*nc*type_size * ny);
#endif

	int pitchrequest = pitch;

	cutilSafeCall(cudaMalloc(device_image,(pitch)*nc*type_size*ny));
	pitchbytes = (pitch)*nc*type_size;

	whole += (int)(pitchbytes * ny);
	if(pitchrequest > 0)
	{
		if(pitchrequest != (pitch))
		{
			fprintf(stderr,"\n\nWARNING: Prior Pitch of %i"
					" is different to Pitch of %i\n\n",pitchrequest,pitch);
		}
	}

#ifdef CUDABASICVERBOSE

	fprintf(stderr,"\t\tAllocated:  %d -> %d   ",
			pitchbytes * ny,whole);

#endif

	return true;
}

void compute_pitch_manual(int nx, int ny, int nc,
									 size_t type_size, int *pitch)
{

	int manualpitch = (int)(nx*nc*type_size);
	while(manualpitch%PADD_SIZE)
	{
		manualpitch += (int)(nc*type_size);
	}
	manualpitch /= (int)(nc*type_size);

	*pitch = manualpitch;

}

int compute_pitch_manual(int nx, int ny, int nc,
									 size_t type_size)
{
	int manualpitch = (int)(nx*nc*type_size);
	while(manualpitch%PADD_SIZE)
	{
		manualpitch += (int)(nc*type_size);
	}
	manualpitch /= (int)(nc*type_size);


	return manualpitch;
}

void compute_pitch_alloc(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int *pitch)
{
	size_t pitchbytes;

	int manualpitch = (int)(nx*nc*type_size);
	while(manualpitch%PADD_SIZE)
	{
		manualpitch += (int)(nc*type_size);
	}
	manualpitch /= (int)(nc*type_size);

	cutilSafeCall( cudaMallocPitch(
			(void**) device_image, &pitchbytes, nx*nc*type_size, ny));
	*pitch = (int)(pitchbytes/(nc*type_size));
	cutilSafeCall(cudaFree(*device_image));

	//Padd Manually
	if((*pitch)*type_size*nc < pitchbytes)
	{
		*pitch = manualpitch;
	}
}

int compute_pitch_alloc(void **device_image, int nx, int ny, int nc,
									 size_t type_size)
{
	size_t pitchbytes;

	int manualpitch = (int)(nx*nc*type_size);
	while(manualpitch%PADD_SIZE)
	{
		manualpitch += (int)(nc*type_size);
	}
	manualpitch /= (int)(nc*type_size);

	int pitch;

	cutilSafeCall( cudaMallocPitch(
			(void**) device_image, &pitchbytes, nx*nc*type_size, ny));
	pitch = (int)(pitchbytes/(nc*type_size));
	cutilSafeCall(cudaFree(*device_image));

	//Padd Manually
	if((pitch)*type_size*nc < pitchbytes)
	{
		pitch = manualpitch;
	}
	return pitch;
}



bool cuda_copy_h2d_2D(const float *host_ptr, float *device_ptr,
											int nx, int ny, int nc,
										  size_t type_size, int pitch)
{
	//fprintf(stderr,"\nCopying %d -> %d : %d   ",
		//	(long)host_ptr, (long)device_ptr,pitch*nc*ny*type_size);
	cutilSafeCall( cudaMemcpy2D(device_ptr, pitch*nc*type_size,
															host_ptr, nx*nc*type_size,
															nx*nc*type_size, ny,
															cudaMemcpyHostToDevice) );
	return true;
}

bool cuda_copy_d2h_2D(const float *device_ptr, float *host_ptr,
											int nx, int ny, int nc,
										  size_t type_size, int pitch)
{
	cutilSafeCall( cudaMemcpy2D(host_ptr, nx*nc*type_size,
															device_ptr, pitch*nc*type_size,
															nx*nc*type_size, ny,
															cudaMemcpyDeviceToHost) );
	return true;
}

bool cuda_copy_d2d(const float *device_in, float *device_out,
											int nx, int ny, int nc,
										  size_t type_size, int pitch)
{
#ifdef CUDABASICVERBOSE
	fprintf(stderr,"\nCopying %d -> %d : %d   ",
			(long)device_in, (long)device_out,pitch*nc*ny*type_size);
#endif
	cutilSafeCall( cudaMemcpy(device_out,device_in,pitch*nc*ny*type_size,
									cudaMemcpyDeviceToDevice) );

	return true;
}

bool cuda_copy_d2d_repitch(const float *device_in, float *device_out,
											int nx, int ny, int nc,
										  size_t type_size, int pitch_in, int pitch_out)
{
	cutilSafeCall( cudaMemcpy2D(device_out,pitch_out*type_size,
			device_in,pitch_in*type_size,nx*nc*type_size,ny,
			cudaMemcpyDeviceToDevice));

	return true;
}

__global__ void set_kernel
(
		float *device_memory,
		int   size,
		float value
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x < size)
	{
		device_memory[x] = value;
	}
}

void cuda_value
(
	float *device_ptr,
	int size,
	float value
)
{
	int nb = size / 256;
	if(nb*256 < size) nb++;

	dim3 dimGrid(nb);
	dim3 dimBlock(256);
	set_kernel<<<dimGrid,dimBlock>>>
			(device_ptr,size,value);
	cutilSafeCall( cudaThreadSynchronize() );
}

void cuda_zero(float *device_ptr, size_t size)
{
	cutilSafeCall(cudaMemset(device_ptr, 0, size));
}


bool cuda_free(void *device_ptr)
{
	cutilSafeCall(cudaFree(device_ptr));
	return true;
}


bool device_query_and_select(int request)
{
  fprintf(stderr," CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
  {
		fprintf(stderr,"cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
		fprintf(stderr,"\nFAILED\n");
		return false;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
  {
     fprintf(stderr,"There is no device supporting CUDA\n");
     return false;
  }

  int dev;
  int driverVersion = 0, runtimeVersion = 0;
  for (dev = 0; dev < deviceCount; ++dev)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0)
    {
    	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
          fprintf(stderr,"There is no device supporting CUDA.\n");
      else if (deviceCount == 1)
          fprintf(stderr,"There is 1 device supporting CUDA\n");
      else
          fprintf(stderr,"There are %d devices supporting CUDA\n", deviceCount);
    }
    fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);

  #if CUDART_VERSION >= 2020
      // Console log
	cudaDriverGetVersion(&driverVersion);
	fprintf(stderr,"  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
	cudaRuntimeGetVersion(&runtimeVersion);
	fprintf(stderr,"  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
  #endif
      fprintf(stderr,"  CUDA Capability Major revision number:         %d\n", deviceProp.major);
      fprintf(stderr,"  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

	fprintf(stderr,"  Total amount of global memory:                 %u bytes\n", (unsigned int)(deviceProp.totalGlobalMem));
  #if CUDART_VERSION >= 2000
      fprintf(stderr,"  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
      //fprintf(stderr,"  Number of cores:                               %d\n", nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount);
  #endif
      fprintf(stderr,"  Total amount of constant memory:               %u bytes\n", (unsigned int)deviceProp.totalConstMem);
      fprintf(stderr,"  Total amount of shared memory per block:       %u bytes\n", (unsigned int)deviceProp.sharedMemPerBlock);
      fprintf(stderr,"  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
      fprintf(stderr,"  Warp size:                                     %d\n", deviceProp.warpSize);
      fprintf(stderr,"  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
      fprintf(stderr,"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
             deviceProp.maxThreadsDim[0],
             deviceProp.maxThreadsDim[1],
             deviceProp.maxThreadsDim[2]);
      fprintf(stderr,"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
             deviceProp.maxGridSize[0],
             deviceProp.maxGridSize[1],
             deviceProp.maxGridSize[2]);
      fprintf(stderr,"  Maximum memory pitch:                          %u bytes\n", (unsigned int)deviceProp.memPitch);
      fprintf(stderr,"  Texture alignment:                             %u bytes\n", (unsigned int)deviceProp.textureAlignment);
      fprintf(stderr,"  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
  #if CUDART_VERSION >= 2000
      fprintf(stderr,"  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
  #endif
  #if CUDART_VERSION >= 2020
      fprintf(stderr,"  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
      fprintf(stderr,"  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
      fprintf(stderr,"  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
      fprintf(stderr,"  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
		                                                            "Default (multiple host threads can use this device simultaneously)" :
	                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																	"Exclusive (only one host thread at a time can use this device)" :
	                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																	"Prohibited (no host thread can use this device)" :
																	"Unknown");
  #endif
}

  // csv masterlog info
  // *****************************
  // exe and CUDA driver name
  fprintf(stderr,"\n");
  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
  char cTemp[10];

  // driver version
  sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, driverVersion%100);
  #else
    sprintf(cTemp, "%d.%d", driverVersion/1000, driverVersion%100);
  #endif
  sProfileString +=  cTemp;

  // Runtime version
  sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
  #else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
  #endif
  sProfileString +=  cTemp;

  // Device count
  sProfileString += ", NumDevs = ";
  #ifdef WIN32
      sprintf_s(cTemp, 10, "%d", deviceCount);
  #else
      sprintf(cTemp, "%d", deviceCount);
  #endif
  sProfileString += cTemp;

  // First 2 device names, if any
  for (dev = 0; dev < ((deviceCount > 2) ? 2 : deviceCount); ++dev)
  {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      sProfileString += ", Device = ";
      sProfileString += deviceProp.name;
  }

  // finish
  fprintf(stderr,"\n\nPASSED\n");
  return true;
}

bool cuda_memory_available
(
	size_t *total,
	size_t *free,
	unsigned int device,
	int verbose
)
{
	int numDevices;
	CUresult res; //, helpRes; // helpRes result of error code as a string
	CUresult helpRes;
	CUdevice dev;
	CUcontext ctx;
	//cuInit(0);

	cuDeviceGetCount(&numDevices);
	if((int)device >= numDevices)
	{
		fprintf(stderr,"\nERROR: Device No %i not available",device);
		return false;
	}
	const char *pStr; // for getting the CUresult string error
	cuDeviceGet(&dev,device);
	cuCtxCreate(&ctx, 0, dev);
	res = cuMemGetInfo(free, total);

	if(res != CUDA_SUCCESS)
	{
		fprintf(stderr,"!!!! cuMemGetInfo failed! (status = %x)", res);
		//helpRes = cuGetErrorString(res, &pStr);
		/*if(helpRes = CUDA_SUCCESS){
			fprintf(stderr, "\nhelpRes success!");
			fprintf(stderr,"\n ERROR: %s", pStr);
		}
		else{
			fprintf(stderr,"\ngetting the string error failed!!");
		}*/
		return false;
	}
	cuCtxDetach(ctx);
	if(verbose) fprintf(stderr,"\nDevice No %i \tFree: %i\tTotal: %i\n",(int)device,(int)(*free),(int)(*total));

	return true;
}
