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

#ifndef MEMDUMP_H
#define MEMDUMP_H

#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

/**
 * Utility class for dumping arbitrary two dimensional opencv matrices to disk and for reading them back in.
 * Contains a method for comparing arbitrary two dimensional opencv matrices.
 */
class MemDump {

public:

	/**
	 * @brief Write a matrix to a binary file
	 * File format:
	 * rows cols type data
	 * @param mat write this to disk
	 * @param file_name create this file and write mat into it or overwritte existing file
	 * @returns true on success, false otherwise
	 */
    static bool writeMatrix(const cv::Mat& mat,const std::string& file_name)
    {
    	using namespace std;
    	ofstream matOut (file_name.c_str(), ios::out | ios::binary);
    	if(!matOut)
    	{
    		return false;
    	}

    	matOut.write((char*)&mat.rows,sizeof(int));
    	matOut.write((char*)&mat.cols,sizeof(int));
    	int type=mat.type();
    	matOut.write((char*)&type,sizeof(int));

		for(int i=0;i<mat.rows;i++)
		{
			for(int j=0;j<mat.cols;j++)
			{
    			matOut.write((char*)&mat.data[i*mat.step[0] + mat.step[1]*j],mat.elemSize());
    		}
    	}
    	matOut.close();
    	return true;
    }

	/**
	 * @brief Read matrix from a binary file that was created using MemDump::writeMatrix(...)
	 * File format:
	 * rows cols type data
	 * @param file_name read from this file
	 * @returns on success the matrix contained in the file, otherwise an emtpy matrix
	 */
    static cv::Mat readMatrix(const std::string& file_name)
    {
    	using namespace std;
    	ifstream matIn(file_name.c_str(), ios::in | ios::binary);
    	if(!matIn)
    	{
    		std::cerr<<"Can't open file "<<file_name.c_str()<<" for reading"<<std::endl;
    		return cv::Mat();
    	}
    	int rows,cols;
    	int type;
    	matIn.read((char*)&rows,sizeof(int));
    	matIn.read((char*)&cols,sizeof(int));
    	matIn.read((char*)&type,sizeof(int));

    	cv::Mat mat(rows,cols,type);

		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
    			if(!matIn.read((char*)&mat.data[i*mat.step[0] + mat.step[1]*j],mat.elemSize()))
    			{
    				std::cerr<<"Can't read from file"<< i<<" "<<j<<std::endl;
    				return cv::Mat();
    			}
    		}
    	}

    	return mat;
    }

    /**
     * Struct used for storing comparison results
     */
    struct ErrorStats {
    	float fractionOfErrorPixels;
    	float averageError;
    	float fractionOfRoundingErrorPixels;
    	float averageRoundingError;
    	int errorPixels;

    	ErrorStats():fractionOfErrorPixels(0),averageError(0),fractionOfRoundingErrorPixels(0),averageRoundingError(0),errorPixels(0)
    	{

    	}


    };

    /**
     * @brief compares arbitrary two dimensional matrix A to two dimensional matrix B with same size and datatype and collects statistics in errorStats
     * @param matA matrix A
     * @param matB matrix B
     * @param roundingErrorThreshold every difference below this threshold will be counted as a rounding error
     * @param errorStats OUT; contains the error statistics on return
     * @returns false if matrices are incompatible
     */
    template<typename T>
    static bool compareMatrices(cv::Mat matA, cv::Mat matB, float roundingErrorThreshold,ErrorStats& errorStats)
    {
    	if(matA.cols!=matB.cols || matA.rows!=matB.rows || matA.type() != matB.type())
    	{
    		return false;
    	}

    	for(int i=0;i<matA.rows;i++)
    	{
    		for(int j=0;j<matB.cols;j++)
    		{
    			T a = *(T*)(matA.data+i*matA.step[0]+matA.step[1]*j);
    			T b = *(T*)(matB.data+i*matB.step[0]+matB.step[1]*j);
    			double diff = fabs(a-b);
//    			std::cout<<*(T*)(matA.data+i*matA.step[0]+matA.step[1]*j)<<" "<<*(T*)(matB.data+i*matB.step[0]+matB.step[1]*j)<<std::endl;
    			if(fabs(a-b)>=roundingErrorThreshold)
    			{
    				errorStats.fractionOfErrorPixels++;
    				errorStats.averageError+=diff;
    				errorStats.errorPixels++;
    			}else if(diff!=0.0)
    			{
    				errorStats.fractionOfRoundingErrorPixels++;
    				errorStats.averageRoundingError+=diff;
    			}
    		}
    	}
    	errorStats.averageError /=(matA.rows*matA.cols);
    	errorStats.averageRoundingError /=(matA.rows*matA.cols);
    	errorStats.fractionOfErrorPixels /=(matA.rows*matA.cols);
    	errorStats.fractionOfRoundingErrorPixels /=(matA.rows*matA.cols);
    }



};

std::ostream& operator <<(std::ostream& out,const MemDump::ErrorStats& stats)
{
	out<<"===== Error statistics ========================"<<std::endl;
	out<<"Absolute Number of Error Pixels: "<<stats.errorPixels<<std::endl;
	out<<"Fraction of Error Pixels:        "<<stats.fractionOfErrorPixels<<std::endl;
	out<<"Average Error:                   "<<stats.averageError<<std::endl;
	out<<"Fraction of rounding errors:     "<<stats.fractionOfRoundingErrorPixels<<std::endl;
	out<<"Average rounding error:          "<<stats.averageRoundingError<<std::endl;
	out<<"==============================================="<<std::endl;
	return out;
}

#endif
