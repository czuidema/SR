/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    flowio.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include "flowio.h"

#include <iostream>
#include <fstream>


bool load_barron_flow_file
(
 std::string filename,
 float *u,
 float *v,
 int   nx,
 int   ny
)
{
	std::fstream file;
	float temp;
	float *utemp;
	float *vtemp;
	int i,j;
	int nxref_and_offsetx;
	int nyref_and_offsety;
	int nxref_no_offsetx;
	int nyref_no_offsety;
	int offsetx;
	int offsety;

	file.open(filename.c_str(),std::ios::in|std::ios::binary);
	if(!file.is_open()){
		fprintf(stderr,"\nERROR: Could not open File %s",filename.c_str());
		return false;
	}

	file.read((char*)(&temp),sizeof(float));
	nxref_and_offsetx  = (int) temp;
	file.read((char*)(&temp),sizeof(float));
	nyref_and_offsety  = (int) temp;
	file.read((char*)(&temp),sizeof(float));
	nxref_no_offsetx  = (int) temp;
	file.read((char*)(&temp),sizeof(float));
	nyref_no_offsety  = (int) temp;
	file.read((char*)(&temp),sizeof(float));
	offsetx = (int) temp;
	file.read((char*)(&temp),sizeof(float));
	offsety = (int) temp;

	if ((nx!=nxref_no_offsetx)||(ny!=nyref_no_offsety)){
		fprintf(stderr,"\nERROR: Wrong Dimensions!");
		return false;
	}

	utemp = new float[nxref_and_offsetx*nyref_and_offsety];
	vtemp = new float[nxref_and_offsetx*nyref_and_offsety];

	for(j=0;j<nyref_and_offsety;j++){
		for(i=0;i<nxref_and_offsetx;i++){
			file.read((char*)(&temp),sizeof(float));
			utemp[j*nxref_and_offsetx+i] = temp;
			file.read((char*)(&temp),sizeof(float));
			vtemp[j*nxref_and_offsetx+i] = temp;
		}
	}

	for(i=0;i<nx;i++){
		for(j=0;j<ny;j++){
			u[j*nx+i] = (float) utemp[(j+offsety)*nxref_and_offsetx+i+offsetx];
			v[j*nx+i] = (float) vtemp[(j+offsety)*nxref_and_offsetx+i+offsetx];
		}
	}

	return true;
}

bool load_flow_file
(
std::string filename,
float *u,
float *v,
int   nx,
int   ny
)
{
	int i;
	std::fstream file;
	float test;

	file.open(filename.c_str(),std::ios::binary|std::ios::in);
	if(!(file.is_open())){
		std::cerr << "\n\nERROR: Could not open File \"" << filename
							<< "\" for reading!";
		return false;
	}
	file.read((char*)(&test),sizeof(float));
	if(test != 202021.25){
		std::cerr << "\n\nERROR: File Fails the Sanity Test!";
		return false;
	}


	int dummy;
	file.read((char*)(&dummy),sizeof(int));
	file.read((char*)(&dummy),sizeof(int));


	for(i = 0; i < nx*ny; i++){
		if(file.eof()){
			std::cerr << "\n\nERROR: File End too soon, at "
					<< i << " of " << (nx*ny);
			return false;
		}
		file.read((char*)(&(u[i])),sizeof(float));
		file.read((char*)(&(v[i])),sizeof(float));
	}

	file.close();

	return true;
}

bool load_flow_file
(
std::string filename,
float  *u,
int    nx,
int    ny
)
{
	std::fstream file;
	float test;

	file.open(filename.c_str(),std::ios::binary|std::ios::in);
	if(!(file.is_open())){
		std::cerr << "\n\nERROR: Could not open File \"" << filename
							<< "\" for reading!";
		return false;
	}
	file.read((char*)(&test),sizeof(float));
	if(test != 202021.25)
	{
		std::cerr << "\n\nERROR: File Fails the Sanity Test!";
		return false;
	}


	int dummy;
	file.read((char*)(&dummy),sizeof(int));
	file.read((char*)(&dummy),sizeof(int));

	fprintf(stderr,"\nReading Flow Data %ix%i",nx,ny);
	file.read((char*)u,nx*ny*sizeof(float2));

	file.close();

	return true;
}


bool load_flow_file
(
std::string filename,
float **u,
float **v,
int   *nx,
int   *ny
)
{
	int i;
	std::fstream file;
	float test;

	file.open(filename.c_str(),std::ios::binary|std::ios::in);
	if(!(file.is_open())){
		std::cerr << "\n\nERROR: Could not open File \"" << filename
							<< "\" for reading!";
		return false;
	}
	file.read((char*)(&test),sizeof(float));
	if(test != 202021.25){
		std::cerr << "\n\nERROR: File Fails the Sanity Test!";
		return false;
	}

	file.read((char*)(nx),sizeof(int));
	file.read((char*)(ny),sizeof(int));

	(*u) = new float[(*nx)*(*ny)];
	(*v) = new float[(*nx)*(*ny)];


	for(i = 0; i < (*nx)*(*ny); i++){
		if(file.eof()){
			std::cerr << "\n\nERROR: File End too soon, at "
					<< i << " of " << (*nx)*(*ny);
			return false;
		}
		file.read((char*)(&((*u)[i])),sizeof(float));
		file.read((char*)(&((*v)[i])),sizeof(float));
	}

	file.close();
	return true;
}


bool load_flow_file
(
std::string filename,
float  **u,
int    *nx,
int    *ny
)
{
	std::fstream file;
	float test;

	file.open(filename.c_str(),std::ios::binary|std::ios::in);
	if(!(file.is_open())){
		std::cerr << "\n\nERROR: Could not open File \"" << filename
							<< "\" for reading!";
		return false;
	}
	file.read((char*)(&test),sizeof(float));
	if(test != 202021.25){
		std::cerr << "\n\nERROR: File Fails the Sanity Test!";
		return false;
	}

	file.read((char*)(nx),sizeof(int));
	file.read((char*)(ny),sizeof(int));

	(*u) = (float*) new float2[(*nx)*(*ny)];

	file.read((char*)(*u),(*nx)*(*ny)*sizeof(float2));

	file.close();
	return true;
}





bool save_flow_file
(
std::string filename,
float *u,
float *v,
int   nx,
int   ny
)
{
	int i;
	std::fstream file;
	file.open(filename.c_str(),std::ios::binary|std::ios::out);
	if(!(file.is_open())){
		std::cerr << "\n\nERROR: Could not open File \"" << filename
							<< "\" for writing!";
		return false;
	}
	file << "PIEH";
	file.write((char*)(&nx),sizeof(int));
	file.write((char*)(&ny),sizeof(int));

	for(i = 0; i < nx*ny; i++){
		file.write((char*)(&(u[i])),sizeof(float));
		file.write((char*)(&(v[i])),sizeof(float));
	}

	file.close();
	return true;
}

bool save_flow_file
(
std::string filename,
float  *u,
int    nx,
int    ny
)
{
	std::fstream file;
	file.open(filename.c_str(),std::ios::binary|std::ios::out);
	if(!(file.is_open())){
		std::cerr << "\n\nERROR: Could not open File \"" << filename
							<< "\" for writing!";
		return false;
	}
	file << "PIEH";
	file.write((char*)(&nx),sizeof(int));
	file.write((char*)(&ny),sizeof(int));

	file.write((char*)u,nx*ny*sizeof(float2));

	file.close();
	return true;
}
