/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    flowio.h
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef FLOWIO_H
#define FLOWIO_H

//#include <cutil.h>
//#include <cutil_inline.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>


bool load_barron_flow_file
(
 std::string filename,
 float *u,
 float *v,
 int   nx,
 int   ny
);

/**Loads a file in the Middlebury ".flo" format into two separate
 * flow arrays u and v, which have been allocated previously with the
 * correct size.
 */
bool load_flow_file
(
std::string filename,
float *u,
float *v,
int   nx,
int   ny
);

/**Loads a file in the Middlebury ".flo" format into one interleaved
 * flow array u, which has been allocated previously with the
 * correct size.
 */
bool load_flow_file
(
std::string filename,
float  *u,
int    nx,
int    ny
);

/**Loads a file in the Middlebury ".flo" format into two separate
 * flow arrays u and v, which is allocated in the function with the
 * correct size.
 */
bool load_flow_file
(
std::string filename,
float **u,
float **v,
int   *nx,
int   *ny
);

/**Loads a file in the Middlebury ".flo" format into one interleaved
 * flow array u, which is allocated in the function with the
 * correct size.
 */
bool load_flow_file
(
std::string filename,
float  **u,
int    *nx,
int    *ny
);

/**Saves two separate flow arrays, one for each direction, into a file
 * of the Middlebury ".flo" format.
 */
bool save_flow_file
(
std::string filename,
float *u,
float *v,
int   nx,
int   ny
);

/**Saves an interleaved flow array into a file
 * of the Middlebury ".flo" format.
 */
bool save_flow_file
(
std::string filename,
float  *u,
int    nx,
int    ny
);



#endif
