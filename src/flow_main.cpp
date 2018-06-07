/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2012 / September 17-24, 2012
*
* project: sorflow
* file:    sorflow_main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <iostream>
#include <flowlib/flowio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <flowlib/flowlib.hpp>

int main(int argc, char *argv[])
{

	/**
	 * Print the Help
	 */
  if (argc < 2) {
    fprintf(stderr,"Usage:""\n%s imageData1 imageData2", argv[0]);
    fprintf(stderr,"   or:""\n%s camera", argv[0]);
    fprintf(stderr, "\n\ngeneral usage further parameters: -option number\n");
    fprintf(stderr, "-l  -lambda \t\t defines lambda [default=5.0]\n");
    fprintf(stderr, "-oi -outeriterations \t defines outeriterations [default=40]\n");
    fprintf(stderr, "-ii -inneriterations \t defines inneriterations [default=2]\n");
    fprintf(stderr, "-sl -startlevel \t defines startlevel [default=3]\n");
    fprintf(stderr, "-el -endlevel \t\t defines endlevel [default=0]\n");
    fprintf(stderr, "-cd  -computationdevice  defines the computation device [default=0], 0 = CPU, 1 = GPU\n");
    fprintf(stderr, "-w  -width \t\t defines camera capture size [default=320]\n");
    fprintf(stderr, "\navailable methods:\n");
    fprintf(stderr,"\n\n");
    return 0;
  }

  /**
   * Standard Parameters
   */
  int start_level = 3;
  int end_level = 0;
  int outer_iterations = 40;
  int inner_iterations = 2;
  float lambda = 0.2f;
  int cameraWidth = 320, cameraHeight = 240;

  int computationDevice = 0;

  std::string partype = "";
  std::string parval = "";

  std::string arg1 = argv[1];

  /**
   * Read Parameters from Input
   */
  int parnum;
  if (arg1 != "camera")
    parnum = 3;
  else
    parnum = 2;
  while(parnum < argc) {
  	partype = argv[parnum++];
  	if(partype == "-l" || partype == "-lambda") {
  		lambda = (float)atof(argv[parnum++]);
  		fprintf(stderr,"\nSmoothness Weight: %f", lambda);
  	}
  	else if(partype == "-oi" || partype == "-outeriterations") {
  		outer_iterations = atoi(argv[parnum++]);
  		fprintf(stderr,"\nOuter Iterations: %i", outer_iterations);
  	}
  	else if(partype == "-ii" || partype == "-inneriterations") {
  		inner_iterations = atoi(argv[parnum++]);
  		fprintf(stderr,"\nInner Iterations: %i", inner_iterations);
  	}
  	else if(partype == "-sl" || partype == "-startlevel") {
  		start_level = atoi(argv[parnum++]);
  		fprintf(stderr,"\nStart Level: %i", start_level);
  	}
  	else if(partype == "-el" || partype == "-endlevel") {
  		end_level = atoi(argv[parnum++]);
  		fprintf(stderr,"\nEnd Level: %i", end_level);
  	}
  	else if ((partype == "-w" || partype == "-width") && strcmp(argv[1], "camera") == 0) {
  		cameraWidth = atoi(argv[parnum++]);
  		cameraHeight = (int)(cameraWidth*3/4);
  		fprintf(stderr,"\nCamera Width: %i\nCamera Height: %i", cameraWidth, cameraHeight);
  	}
  	else if(partype == "-cd"|| partype == "-computationdevice"){
  		std::string parval = argv[parnum++];
  		if(parval == "0")computationDevice = 0;
  		if(parval == "1")computationDevice = 1;
  		fprintf(stderr,"\nComputation Device: %s",computationDevice ? "GPU" : "CPU");
  	}
  }

  if (arg1 != "camera" && arg1 != "video") {

    cv::Mat img1 = cv::imread(argv[1], 0); // 0 = force the image to be grayscale
    cv::Mat img2 = cv::imread(argv[2], 0);

    if (img1.rows != img2.rows || img1.cols != img2.cols) {
    	fprintf(stderr,
    			"\nError: Image formats are not equal: (%i|%i) vs. (%i|%i)",
    			img1.cols,img1.rows,img2.cols,img2.rows);
      return 1;
    }

    fprintf(stderr,"\nInput Image Dimensions: (%i|%i)",img1.cols,img1.rows);

    img1.convertTo(img1,CV_32FC1);
    img2.convertTo(img2,CV_32FC1);

    fprintf(stderr,"\nCreating FlowLib Object");

    FlowLib *flowlib = NULL;
    if(computationDevice == 1)
    	flowlib = new FlowLibGpuSOR(img1.cols,img1.rows);
    else if(computationDevice == 0)
    	flowlib = new FlowLibCpuSOR(img1.cols,img1.rows);


    fprintf(stderr,"\nSetting up Flow");
    flowlib->setLambda(lambda);
		flowlib->setOuterIterations(outer_iterations);
		flowlib->setInnerIterations(inner_iterations);
		flowlib->setStartLevel(start_level);
		flowlib->setEndLevel(end_level);

    flowlib->setInput((float*)img1.data);
    flowlib->setInput((float*)img2.data);

    fprintf(stderr,"\nComputing Flow");

    flowlib->computeFlow();


    float *u = new float[img1.cols*img1.rows*2];
    float *v = new float[img1.cols*img1.rows];
    flowlib->getOutput(u,v);
    for(int x=img1.cols*img1.rows-1;x>=0;x--){
    	u[2*x] = u[x];
    	u[2*x+1] = v[x];
    }
    save_flow_file("output.flo",u, img1.cols, img1.rows);
    delete [] u;

    fprintf(stderr,"\nDisplaying Output");

    cv::Mat outputImg(cv::Size(img1.cols,img1.rows),CV_32FC3);

    flowlib->getOutputBGR((float*)outputImg.data,0.0f,1.0f,0.0f,1.0f);


    cv::imshow("Input Image 1",img1/255.0f);
    cv::imshow("Input Image 2",img2/255.0f);
    cv::imshow("Output Image",outputImg);

    cv::imwrite(computationDevice ? "flowImageGPU.png" : "flowImageCPU.png",outputImg*255.0f);

    cvMoveWindow("Input Image 1", 100, 100);
    cvMoveWindow("Input Image 2", 450, 100);
    cvMoveWindow("Output Image", 100, 400);

    printf("\nPress Esc on the image to exit...\n");
    cv::waitKey(0);

    fprintf(stderr,"\nDeleting Flow Object");
    delete flowlib;
    fprintf(stderr,"\nFlow Object Deleted.");

  } // endif image file

  //----------------------------------------------------------------------------
  // Camera Mode
  //----------------------------------------------------------------------------
  else{

  	cv::VideoCapture *capture = 0;
  	bool videomode = arg1 == "video";
  	if(videomode){
    	fprintf(stderr,"\nReading Images from Video %s",argv[2]);
  		capture = new cv::VideoCapture(argv[2]);
  	}
  	else{
    	fprintf(stderr,"\nReading Images from Camera");
    	capture = new cv::VideoCapture(0);
  	}
  	if(!capture->isOpened()){
  		std::cerr << std::endl << "Could not Open Capture Device";
  		return -1;
  	}
  	if(!videomode){
  		fprintf(stderr,"\nSetting Camera up to (%i|%i)",cameraWidth,cameraHeight);
			capture->set(CV_CAP_PROP_FRAME_WIDTH,cameraWidth);
			capture->set(CV_CAP_PROP_FRAME_HEIGHT,cameraHeight);
  	}
  	cv::Mat *img1 = new cv::Mat();
  	cv::Mat *img2 = new cv::Mat();



  	(*capture) >> (*img1);
  	fprintf(stderr,"\nImage Dimensions: (%i|%i)",img1->cols,img1->rows);

    float *u = new float[img1->cols*img1->rows*2];
    float *v = new float[img1->cols*img1->rows];

    fprintf(stderr,"\nCreating FlowLib Object");
    FlowLib *flowlib = NULL;
    if(computationDevice == 1)
    	flowlib = new FlowLibGpuSOR(img1->cols,img1->rows);
    else if(computationDevice == 0)
    	flowlib = new FlowLibCpuSOR(img1->cols,img1->rows);
    fprintf(stderr,"\nFlowLib Object created.");

    fprintf(stderr,"\nSetting up Flow");

    flowlib->setLambda(lambda);
		flowlib->setOuterIterations(outer_iterations);
		flowlib->setInnerIterations(inner_iterations);
		flowlib->setStartLevel(start_level);
		flowlib->setEndLevel(end_level);

    fprintf(stderr,"\nFlow Setup Finished");

  	cv::namedWindow("Input Image",1);
  	cv::namedWindow("Flow",1);

    cv::Mat outputImg(cv::Size(img1->cols,img1->rows),CV_32FC3);

//    int counter = 0;

  	while (cv::waitKey(30) < 0
  			){
  		(*capture) >> (*img1);
  		cv::cvtColor(*img1,*img1,CV_BGR2GRAY);
  		img1->convertTo(*img1,CV_32FC1);
  		flowlib->setInput((float*)img1->data);
      printf("\n%f fps",1000.0f/flowlib->computeFlow());
      flowlib->getOutputBGR((float*)outputImg.data,0.0f,5.0f,0.0f,1.0f);

  		cv::imshow("Input Image",(*img1)/255.0f);
  		cv::imshow("Flow",outputImg);
      cv::Mat *temp = img1; img1 = img2; img2 = temp;
  	}
  	delete capture;

  	delete [] u;
  	delete [] v;

  } // endif camera

  fprintf(stderr,"\n\n");
  return 0;
}
