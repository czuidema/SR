/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    cuda_basic.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef FILLIB_BASIC_CUH
#define FILLIB_BASIC_CUH

//#include <cutil.h>
//#include <cutil_inline.h>

#include<stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cutil_math.h>


#define SAFE_MODE
#ifdef SAFE_MODE
#define catchkernel cutilSafeCall(cudaThreadSynchronize())
#else
#define catchkernel
#endif



#ifdef SM_13
#define ATOMICADD(a,b) atomicAdd(((a)),((b)))
#else
#define ATOMICADD(a,b) *((a)) += ((b))
#endif

#ifndef ABS
//#define ABS(a) ( ( (a) < 0 ) ? ( - (a) ) : (a) )
#define ABS(a) fabsf((a))
#endif

#ifndef DIVCEIL_HOST
#define DIVCEIL_HOST
inline int divceil(int a, int b)
{
	return (a%b == 0) ? (a/b) : (a/b + 1);
}
#endif

//#define SF_TEXTURE_OFFSET 0.0f

#ifndef cutilSafeCall
#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
  if(cudaSuccess != err) {
    fprintf(stderr,"%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}
#endif

bool cuda_malloc2D(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int *pitch);

bool cuda_malloc2D_manual(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int pitch);

void compute_pitch_manual(int nx, int ny, int nc,
									 size_t type_size, int *pitch);

int compute_pitch_manual(int nx, int ny, int nc,
									 size_t type_size);

void compute_pitch_alloc(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int *pitch);

int compute_pitch_alloc(void **device_image, int nx, int ny, int nc,
									 size_t type_size);

bool cuda_copy_h2d_2D(const float *host_ptr, float *device_ptr,
											int nx, int ny, int nc,
										  size_t type_size, int pitch);

bool cuda_copy_d2h_2D(const float *device_ptr, float *host_ptr,
											int nx, int ny, int nc,
										  size_t type_size, int pitch);

bool cuda_copy_d2d(const float *device_in, float *device_out,
											int nx, int ny, int nc,
										  size_t type_size, int pitch);

bool cuda_copy_d2d_repitch(const float *device_in, float *device_out,
											int nx, int ny, int nc,
										  size_t type_size, int pitch_in, int pitch_out);

void cuda_zero(float *device_ptr, size_t size);

void cuda_value
(
	float *device_prt,
	int size,
	float value
);

bool cuda_free(void *device_ptr);


bool device_query_and_select(int request);

bool cuda_memory_available
(
	size_t *total,
	size_t *free,
	unsigned int device = 0,
	int verbose = 0
);








#endif // #ifndef FILLIB_BASIC_CUH


