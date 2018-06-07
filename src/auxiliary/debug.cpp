/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    debug.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * cuda_debug.cpp
 *
 *  Created on: Apr 18, 2012
 *      Author: steinbrf
 */

#include <auxiliary/cuda_basic.cuh>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#ifdef CIMGDEBUG
#include "CImg.h"
using namespace cimg_library;
#endif

void saveFloatImage
(
		std::string savename,
		float *image,
		int   width,
		int   height,
		int   channels,
		float minimum,
		float maximum
)
{
	cv::Mat outImage(cv::Size(width,height),((channels==3) ? CV_32FC3 : CV_32FC1));
	for(int x=0;x<width*height*channels;x++) ((float*)(outImage.data))[x] = image[x];
	if(channels==3){
		for(int x=0;x<width*height;x++){
			float temp = ((float*)(outImage.data))[x*3];
			((float*)(outImage.data))[x*3] = ((float*)(outImage.data))[x*3+2];
			((float*)(outImage.data))[x*3+2] = temp;
		}
	}

	if(maximum<minimum){
		float avg = 0.0f;
		float newmax = -1e6f;
		float newmin = 1e6f;
		for(int x=0;x<width*height*channels;x++){
			avg += ((float*)(outImage.data))[x];
			if(newmax < ((float*)(outImage.data))[x]) newmax = ((float*)(outImage.data))[x];
			if(newmin > ((float*)(outImage.data))[x]) newmin = ((float*)(outImage.data))[x];
		}
		avg /= (float)(width*height*channels);
		maximum = newmax;
		if(minimum > 0.0f) {
			minimum = newmin;
		}
		fprintf(stderr,"\nDebug Image %s Minimum: %f Maximum: %f Average: %f",savename.c_str(),minimum,maximum,avg);
	}

	cv::imwrite(savename.c_str(),(outImage-minimum)*(255.0f/(maximum-minimum)));
}

void showFloatImage
(
		std::string showname,
		float *image,
		int   width,
		int   height,
		int   channels,
		float minimum,
		float maximum
)
{
	cv::Mat outImage(cv::Size(width,height),((channels==3) ? CV_32FC3 : CV_32FC1));
	for(int x=0;x<width*height*channels;x++) ((float*)(outImage.data))[x] = image[x];
	if(channels==3){
		for(int x=0;x<width*height;x++){
			float temp = ((float*)(outImage.data))[x*3];
			((float*)(outImage.data))[x*3] = ((float*)(outImage.data))[x*3+2];
			((float*)(outImage.data))[x*3+2] = temp;
		}
	}

	if(maximum<minimum){
		float avg = 0.0f;
		float newmax = -1e6f;
		float newmin = 1e6f;
		for(int x=0;x<width*height*channels;x++){
			avg += ((float*)(outImage.data))[x];
			if(newmax < ((float*)(outImage.data))[x]) newmax = ((float*)(outImage.data))[x];
			if(newmin > ((float*)(outImage.data))[x]) newmin = ((float*)(outImage.data))[x];
		}
		avg /= (float)(width*height*channels);
		maximum = newmax;
		if(minimum > 0.0f) {
			minimum = newmin;
		}
		fprintf(stderr,"\nDebug Image %s Minimum: %f Maximum: %f Average: %f",showname.c_str(),minimum,maximum,avg);
	}

	cv::imshow(showname.c_str(),(outImage-minimum)*(1.0f/(maximum-minimum)));
}

void saveCudaImage
(
		std::string savename,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum,
		float maximum
)
{
	cv::Mat outImage(cv::Size(width,height),((channels==3) ? CV_32FC3 : CV_32FC1));
	cuda_copy_d2h_2D(image,(float*)outImage.data,width,height,channels,sizeof(float),pitch);
	if(channels==3){
		for(int x=0;x<width*height;x++){
			float temp = ((float*)(outImage.data))[x*3];
			((float*)(outImage.data))[x*3] = ((float*)(outImage.data))[x*3+2];
			((float*)(outImage.data))[x*3+2] = temp;
		}
	}

	if(maximum<minimum){
		float avg = 0.0f;
		float newmax = -1e6f;
		float newmin = 1e6f;
		for(int x=0;x<width*height*channels;x++){
			avg += ((float*)(outImage.data))[x];
			if(newmax < ((float*)(outImage.data))[x]) newmax = ((float*)(outImage.data))[x];
			if(newmin > ((float*)(outImage.data))[x]) newmin = ((float*)(outImage.data))[x];
		}
		avg /= (float)(width*height*channels);
		maximum = newmax;
		if(minimum > 0.0f) {
			minimum = newmin;
		}
		fprintf(stderr,"\nDebug Image %s Minimum: %f Maximum: %f Average: %f",savename.c_str(),minimum,maximum,avg);
	}

	cv::imwrite(savename.c_str(),(outImage-minimum)*(255.0f/(maximum-minimum)));
}

void showCudaImage
(
		std::string showname,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum,
		float maximum
)
{
	cv::Mat outImage(cv::Size(width,height),((channels==3) ? CV_32FC3 : CV_32FC1));
	cuda_copy_d2h_2D(image,(float*)outImage.data,width,height,channels,sizeof(float),pitch);
	if(channels==3){
		for(int x=0;x<width*height;x++){
			float temp = ((float*)(outImage.data))[x*3];
			((float*)(outImage.data))[x*3] = ((float*)(outImage.data))[x*3+2];
			((float*)(outImage.data))[x*3+2] = temp;
		}
	}

	if(maximum<minimum){
		float avg = 0.0f;
		float newmax = -1e6f;
		float newmin = 1e6f;
		for(int x=0;x<width*height*channels;x++){
			avg += ((float*)(outImage.data))[x];
			if(newmax < ((float*)(outImage.data))[x]) newmax = ((float*)(outImage.data))[x];
			if(newmin > ((float*)(outImage.data))[x]) newmin = ((float*)(outImage.data))[x];
		}
		avg /= (float)(width*height*channels);
		maximum = newmax;
		if(minimum > 0.0f) {
			minimum = newmin;
		}
		fprintf(stderr,"\nDebug Image %s Minimum: %f Maximum: %f Average: %f",showname.c_str(),minimum,maximum,avg);
	}

	cv::imshow(showname.c_str(),(outImage-minimum)*(1.0f/(maximum-minimum)));
}

void saveCudaImageReciprocal
(
		std::string savename,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum,
		float maximum
)
{
	cv::Mat outImage(cv::Size(width,height),((channels==3) ? CV_32FC3 : CV_32FC1));
	cuda_copy_d2h_2D(image,(float*)outImage.data,width,height,channels,sizeof(float),pitch);

	for(int x=0;x<width*height*channels;x++){
		((float*)(outImage.data))[x]  = (((float*)(outImage.data))[x] > 0.0f)
			? 1.0f/((float*)(outImage.data))[x] : 0.0f;
	}

	if(channels==3){
		for(int x=0;x<width*height;x++){
			float temp = ((float*)(outImage.data))[x*3];
			((float*)(outImage.data))[x*3] = ((float*)(outImage.data))[x*3+2];
			((float*)(outImage.data))[x*3+2] = temp;
		}
	}

	if(maximum<minimum){
		float newmax = -1e6f;
		for(int x=0;x<width*height*channels;x++){
			if(newmax <((float*)(outImage.data))[x]) newmax = ((float*)(outImage.data))[x];
		}
		fprintf(stderr,"\nDebug Image Inverse %s Maximum: %f",savename.c_str(),newmax);
		maximum = newmax;
	}

	cv::imwrite(savename.c_str(),(outImage-minimum)*(255.0f/(maximum-minimum)));
}

void showCimg
(
		std::string showname,
		float *image,
		int   width,
		int   height,
		int   pitch,
		int   channels,
		float minimum,
		float maximum
)
{

	float *image_cpu = new float[width*height];
	cuda_copy_d2h_2D(image,image_cpu,width,height,channels,sizeof(float),pitch);

#ifdef CIMGDEBUG
	CImg<float> im;
	im.assign(image_cpu, width, height);
	im.display();
#endif

	delete [] image_cpu;
}
