/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    MemDump.hpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 *  MemDump.cpp
 *
 *  Created on: Oct 19, 2012
 *      Author: seidelf
 */

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <memdump/MemDump.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

	//std::cout<<cv::DataType<cv::Vec3f>::type<<std::endl;
	cv::Vec3f val;
	val[0]=0.0;
	val[1]=0.0;
	val[2]=0.0;
	cv::Mat_<cv::Vec3f> a(128,64,val);

	cv::Mat_<cv::Vec3f> b(128,64,val);
	b(1,3).val[0]=1E-3;
	b(1,7).val[1]=1E-5;
	b(127,54).val[2]=1E-1;
	b(105,63).val[0]=1E-7;

	MemDump::writeMatrix(a,"a.mat");
	MemDump::writeMatrix(b,"b.mat");


}
