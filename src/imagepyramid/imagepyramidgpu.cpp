/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    imagepyramidgpu.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

#include "imagepyramid.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <auxiliary/cuda_basic.cuh>

#include <string>
#include <math.h>

#include <linearoperations/linearoperations.cuh>

ImagePyramidGpu::ImagePyramidGpu():
ImagePyramid(),
pitch(0),
_help_g(NULL)
{

}

ImagePyramidGpu::ImagePyramidGpu
(
	int par_nx,
	int par_ny,
	int par_nc,
	float par_scalefactor,
	float *par_first_level,
	int   par_nx_min,
	int   par_ny_min
)
: ImagePyramid(par_nx,par_ny,par_nc,par_scalefactor,par_nx_min,par_ny_min)
{


	int rwidth = par_nx;
	int rheight = par_ny;
	nc = par_nc;

	nl = (rwidth > par_nx_min && rheight > par_ny_min) ? 1 : 0;

	if(nc != 1 && nc != 2 && nc != 4){
		fprintf(stderr,"\nERROR: Only Pyramids of 1, 2, and 4 Channels are supported!");
		exit(1);
	}

	while(rwidth > par_nx_min && rheight > par_ny_min){
		rwidth = (int)(ceil((float)rwidth*scalefactor));
		rheight = (int)(ceil((float)rheight*scalefactor));
		nl++;
	}


	if(nl == 0){
		fprintf(stderr,"\n\nERROR: Images are too small to build Pyramide: %d %d %d ",
				rwidth, rheight, nc);
		exit(1);
	}


	level =  (float**)(malloc(nl*sizeof(float*)));
	nx =  (int*)   (malloc(nl*sizeof(int)));
	ny = (int*)   (malloc(nl*sizeof(int)));
	pitch =  (int*)   (malloc(nl*sizeof(int)));

	rwidth = par_nx;
	rheight = par_ny;
	fprintf(stderr,"\n%d Level Pyramide:",nl);
	for(int i=0;i<nl;i++){
		fprintf(stderr," (%d,%d) ",rwidth,rheight);
		nx[i] = rwidth;
		ny[i] = rheight;
		pitch[i] = -1;
		cuda_malloc2D((void**)&(level[i]),rwidth,rheight,nc,sizeof(float),&(pitch[i]));

		rwidth = (int)(ceil((float)rwidth*scalefactor));
		rheight = (int)(ceil((float)rheight*scalefactor));
	}

	if(par_first_level){
		cuda_copy_h2d_2D(par_first_level,level[0],
											nx[0],ny[0],1,sizeof(float),pitch[0]);
	}
	cuda_malloc2D((void**)&(_help_g),nx[0],ny[0],nc,sizeof(float),&(pitch[0]));
}

ImagePyramidGpu::~ImagePyramidGpu()
{
	if(level){
		for(int i=0;i<nl;i++){
			cuda_free(level[i]);
		}
	}
	if(level) delete level;
	if(nx) delete nx;
	if(ny) delete ny;
	if(pitch) delete pitch;
	if(_help_g) cuda_free(_help_g);
}

bool ImagePyramidGpu::fill()
{
	int i;
	if(nc != 1)	{
		fprintf(stderr,"\nERROR: Downsampling only supported for 1-channel images!");
		return false;
	}
	for(i=0;i<nl-1;i++){
		resampleAreaParallelSeparate(level[i],level[i+1],nx[i],ny[i],pitch[i],nx[i+1],ny[i+1],pitch[i+1],_help_g);
	}

	return true;
}

bool ImagePyramidGpu::fill(int start_level, int end_level)
{
	if(end_level > nl - 1) end_level = nl-1;
	int i;
	if(nc != 1){
		fprintf(stderr,"\nERROR: Downsampling only supported for 1-channel images!");
		return false;
	}
	for(i=start_level;i<end_level;i++){
		resampleAreaParallelSeparate(level[i],level[i+1],nx[i],ny[i],pitch[i],nx[i+1],ny[i+1],pitch[i+1],_help_g);
	}
	return true;
}


bool ImagePyramidGpu::resample_area(int source_level, int target_level)
{
	if(source_level >= nl || source_level < 0
			 || target_level >= nl || target_level < 0){
		fprintf(stderr,"\nERROR: GpuPyramid::upsample - Level does not exist!");
		return false;
	}


	if(nc == 1){
		resampleAreaParallelSeparate(level[source_level],level[target_level],
				nx[source_level],ny[source_level],pitch[source_level],
				nx[target_level],ny[target_level],pitch[target_level],_help_g);
	}
	else{
		fprintf(stderr,"\nResampling for more than one channel not supported!");
	}
	return true;
}








