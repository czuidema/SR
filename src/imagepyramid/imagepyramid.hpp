/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    imagepyramid.hpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * imagepyramid.hpp
 *
 *  Created on: Mar 15, 2012
 *      Author: steinbrf
 */

#ifndef IMAGEPYRAMIDCPU_HPP_
#define IMAGEPYRAMIDCPU_HPP_

#ifndef NX_MIN
#define NX_MIN 4
#endif
#ifndef NY_MIN
#define NY_MIN 4
#endif


#define DOWNSAMPLETYPE DS_WEDEL

/*!
 * Base class for an image pyramid
 */
class ImagePyramid
{
public:
	//! Constructor
	ImagePyramid();
	//! Copy Constructor
	ImagePyramid(const ImagePyramid &other);
	//! Parametrized Constructor
	ImagePyramid(int par_nx, int par_ny, int par_nc = 1,
							 float par_scalefactor = 0.5f,
								int par_nx_min = NX_MIN,
								int par_ny_min = NY_MIN);
	//! Destructor
	~ImagePyramid();
	//! Fills the Image Pyramid from first level to maximum level
	virtual bool fill() = 0;
	//! Fills the Image Pyramid from specified start level to specified end level
	virtual bool fill(int start_level, int end_level) = 0;

	//! Number of Levels
	int nl;
	//! Memory for each level
	float **level;
	//! Width of each level
	int *nx;
	//! Height of each level
	int *ny;
	//! Number of channels
	int nc;
	//! Scale factor for the pyramid
	float scalefactor;
	//! Minimum width
	int nx_min;
	//! Minimum height
	int ny_min;
protected:
};

/*!
 * CPU Version of the image pyramid
 */
class ImagePyramidCpu : public ImagePyramid
{
public:
	//! Constructor
	ImagePyramidCpu();
	//! Copy Constructor
	ImagePyramidCpu(const ImagePyramidCpu& other);
	//! Parametrized Constructor
	ImagePyramidCpu(int par_nx, int par_ny, int par_nc = 1,
							 float par_scalefactor = 0.5f,
								int par_nx_min = NX_MIN,
								int par_ny_min = NY_MIN);
	//! Assignment operator
	ImagePyramidCpu & operator=(const ImagePyramidCpu &other);
	//! Destructor
	~ImagePyramidCpu();
	//! Fills the Image Pyramid from first level to maximum level
	bool fill();
	//! Fills the Image Pyramid from specified start level to specified end level
	bool fill(int start_level, int end_level);

protected:
	//! Self explainatory
	void compute_and_allocate_levels(int par_nx, int par_ny);
	//! Helper array
	float *help;
};

/*!
 * GPU Version of the image pyramid
 */
class ImagePyramidGpu : public ImagePyramid
{
public:
	//! Constructor
	ImagePyramidGpu();
	//! Parametrized Constructor
	ImagePyramidGpu(int par_width, int par_height,
						 int par_channels = 1,
			 	 	 	 float par_scalefactor = 0.5f,
			 	 	 	 float *par_first_level = 0,
			 	 	 	 int   par_nx_min = 4,
			 	 	   int   par_ny_min = 4);
	//! Destructor
	~ImagePyramidGpu();

	//! Fills the Image Pyramid from first level to maximum level
	bool fill();
	//! Fills the Image Pyramid from specified start level to specified end level
	bool fill(int start_level, int end_level);

	//! Pitch (padded width) of the GPU fields
	int *pitch;

protected:
	//! Helper function for resampling
	bool resample_area(int source_level, int target_level);
	//! Helper array
	float *_help_g;
};



#endif /* IMAGEPYRAMIDCPU_HPP_ */
