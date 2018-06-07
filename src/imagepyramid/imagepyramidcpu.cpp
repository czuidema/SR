/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    imagepyramidcpu.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

#include "imagepyramid.hpp"
#include <math.h>
#include <linearoperations/linearoperations.h>
#include <stdio.h>
#include <string.h>
#include <iostream>


#define DEBUG
#ifdef DEBUG
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


using namespace std;

ImagePyramid::ImagePyramid()
: nl(-1), level(0), nx(0), ny(0), nc(-1), scalefactor(0.0f), nx_min(NX_MIN), ny_min(NY_MIN)
{}

ImagePyramid::~ImagePyramid(){}

ImagePyramid::ImagePyramid(int par_nx, int par_ny, int par_nc,
		float par_scalefactor, int par_nx_min, int par_ny_min)
: nc(par_nc), scalefactor(par_scalefactor), nx_min(par_nx_min), ny_min(par_ny_min)
{}

ImagePyramidCpu::ImagePyramidCpu()
: help(0)
{}

ImagePyramidCpu::ImagePyramidCpu(const ImagePyramidCpu& other) {
	*this = other;
}

ImagePyramidCpu& ImagePyramidCpu::operator=(const ImagePyramidCpu &other) {
	nl = other.nl;
	nc = other.nc;
	scalefactor = other.scalefactor;
	nx_min = other.nx_min;
	ny_min = other.ny_min;

	nx = new int[nl];
	ny = new int[nl];
	level = new float*[nl];
	level[0] = other.level[0];
	for(int i=0;i<nl;i++) {
		nx[i] = other.nx[i];
		ny[i] = other.ny[i];
		if(i>0) {
			level[i] = new float[nx[i]*ny[i]*nc];
			memcpy(level[i],other.level[i],nx[i]*ny[i]*nc*sizeof(float));
		}
	}

	help = (other.help) ? new float[nx[0]*ny[0]*nc] : 0;
	return *this;
}

ImagePyramidCpu::ImagePyramidCpu
(
	int par_nx,
	int par_ny,
	int par_nc,
	float par_scalefactor,
	int par_nx_min,
	int par_ny_min
)
: ImagePyramid(par_nx,par_ny,par_nc,par_scalefactor,par_nx_min,par_ny_min),
help(0)
{
	compute_and_allocate_levels(par_nx,par_ny);
}

ImagePyramidCpu::~ImagePyramidCpu()
{
	if(nx && ny && level && nl > 0)
	{
		for(int i=1;i<nl;i++)
		{
			delete [] level[i];
		}
		delete [] nx;
		delete [] ny;
		delete [] level;
	}
	if(help) delete [] help;
}

bool ImagePyramidCpu::fill()
{
	if(!level){
		fprintf(stderr,"\nERROR: Cannot fill a Pyramid which is not allcoated!");
		return false;
	}

	if(!(level[0])){
		fprintf(stderr,"\nERROR: Input Image is not set in Level 0!");
		return false;
	}

	for(int i=0;i<nl-1;i++){
		resampleAreaParallelizableSeparate(level[i],level[i+1],nx[i],ny[i],nx[i+1],ny[i+1],help);
	}

	return true;
}

bool ImagePyramidCpu::fill(int start_level, int end_level)
{
	if(start_level <= end_level){
		fprintf(stderr,"\nERROR: start_level <= end_level!");
		return false;
	}
	if(!level){
		fprintf(stderr,"\nERROR: Cannot fill a Pyramid which is not allcoated!");
		return false;
	}

	if(!(level[0])){
		fprintf(stderr,"\nERROR: Input Image is not set in Level 0!");
		return false;
	}

	if(end_level != 0){
		resampleAreaParallelizableSeparate(level[0],level[end_level],nx[0],ny[0],nx[end_level],ny[end_level],help);
	}
	for(int i=end_level;i<start_level-1;i++){
		resampleAreaParallelizableSeparate(level[i],level[i+1],nx[i],ny[i],nx[i+1],ny[i+1],help);
	}

	return true;
}

void ImagePyramidCpu::compute_and_allocate_levels(int par_nx, int par_ny)
{
	if(par_nx < nx_min || par_ny < ny_min){
		nl = 0;
		return;
	}

	nl = 1;
	int rnx = par_nx;
	int rny = par_ny;

	while(rnx >= nx_min && rny >= ny_min){
		rnx = (int)(ceil((float)rnx*scalefactor));
		rny = (int)(ceil((float)rny*scalefactor));
		nl++;
	}
	nl--;

	nx = new int[nl];
	ny = new int[nl];
	level = new float*[nl];

	rnx = par_nx;
	rny = par_ny;

	int i = 0;

	while(rnx >= nx_min && rny >= ny_min){
		nx[i] = rnx;
		ny[i++] = rny;

		rnx = (int)(ceil((float)rnx*scalefactor));
		rny = (int)(ceil((float)rny*scalefactor));
	}

	level[0] = 0;
	for(int i=1;i<nl;i++){
		level[i] = new float[nx[i]*ny[i]*nc];
	}
	help = new float[nx[0]*ny[0]*nc];

}




