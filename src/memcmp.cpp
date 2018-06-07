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

	if(argc<4)
	{
		cerr<<"Usage: memcmp fileA.dump fileB.dump absDiffThreshold"<<endl;
		exit(0);
	}

	string fileA=argv[1];
	string fileB=argv[2];
	double absDiffThreshold=atof(argv[3]);
	if(absDiffThreshold==0)
	{
		std::cerr<<"Absolute difference threshold has to be > 0"<<std::endl;
		exit(1);
	}

	cv::Mat matA=MemDump::readMatrix(fileA);
	cv::Mat matB=MemDump::readMatrix(fileB);

	if(matA.empty() || matB.empty())
	{
		std::cerr<<"Error reading matrix"<<std::endl;
	}
//	cv::imshow("A",matA);
//	cv::imshow("B",matB);
//	cv::waitKey(3000);

	MemDump::ErrorStats stats;

	MemDump::compareMatrices<float>(matA,matB,absDiffThreshold,stats);

	std::cout<<stats;


}



