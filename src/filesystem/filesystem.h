/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: superresolution
* file:    filesystem.h
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/

/*
 * filesystem.h
 *
 *  Created on: Feb 29, 2012
 *      Author: steinbrf
 */

#ifndef FILESYSTEM_H_
#define FILESYSTEM_H_


#include <dirent.h>
#include <string>
#include <vector>
#include <list>

/*!
 * Class for computing and storing information about a folder in the file system
 */
class Folder
{
public:
	//! Constructor
	Folder();
	//! Constructor for a specified path
	Folder(std::string p_path);
	//! Destructor
	~Folder();
	//! Loads the folder contents a specified folder
	bool load(std::string p_path);
	//! Number of files in the folder
	size_t size();
	//! Returns all filenames with a specified ending in the folder as a list
	std::list<std::string> getFilesWithEndingList(std::string ending);
	//! Returns all filenames with a specified ending in the folder as a vector
	std::vector<std::string> getFilesWithEndingVector(std::string ending);
	//! Returns whether a folder exists
	bool is_valid();
	//! Returns whether a file in the folder exists
	bool fileExists(std::string filename, std::string prefix = "");

	//! Stores the path to the folder
	std::string path;
	//! Stores all files in the folder
	std::vector<std::string> files;
private:
};

#endif /* FILESYSTEM_H_ */
