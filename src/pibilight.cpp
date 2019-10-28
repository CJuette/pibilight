#include "pibilight.h"
//#include "yaml-cpp/yaml.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>

using namespace std;
using namespace cv;

string logFilePrefix = "/home/pi/pibilight/log/";
string logFileName;
bool logCreated = false;
int cameraFD;
int outputFD;
int width = 720;
int height = 480;
int outWidth = 640;
int outHeight = 360;
constexpr int BYTES_PER_PIXEL_OUT = 4;
size_t bufferSize = outWidth * outHeight * BYTES_PER_PIXEL_OUT;

#define CONFIG_FILE "/home/pi/pibilight/config.yml"

string loopbackDevice = "/dev/video0";
string cameraDevice = "/dev/video1";

unsigned char * outBuffer;

VideoCapture cap;

Mat currentImage;
Mat processedImage;
Mat undistortedTemp;
Mat bgraImg;
Mat rgbaImg;

Mat cameraMatrix, distortionCoefficients;

Point2f sourcePoints[4];
Point2f destinationPoints[4];

Mat perspTransform;

bool createLogFile()
{
	time_t currentTime = time(nullptr);
	char timeStr[100];
	strftime(timeStr, sizeof(timeStr), "%Y-%m-%d_%H-%M", localtime(&currentTime));
	logFileName = timeStr;
	logFileName.append(".log");

	ofstream outfile(logFilePrefix + logFileName);
	outfile << "Log created on " << string(timeStr) << endl << flush;
	outfile.close();

	return true;
}

void logLine(string line)
{
	if(!logCreated)
	{
		logCreated = createLogFile();
	}

	ofstream file;
	file.open(logFilePrefix+logFileName, ios_base::app);
	file << line << endl;
	file.close();

	cout << line << endl;
}

void loadConfig()
{
	logLine("Loading config file.");
	FileStorage configFile(CONFIG_FILE, FileStorage::READ);

	if(!configFile["width"].empty())
	{
		width = (int)configFile["width"];
	}

	if(!configFile["height"].empty())
	{
		height = (int)configFile["height"];
	}

	if(!configFile["outWidth"].empty())
	{
		outWidth = (int)configFile["outWidth"];
	}

	if(!configFile["outHeight"].empty())
	{
		outHeight = (int)configFile["outHeight"];
	}

	if(!configFile["camera_device"].empty())
	{
		cameraDevice = (string)configFile["camera_device"];
	}

	if(!configFile["loopback_device"].empty())
	{
		loopbackDevice = (string)configFile["loopback_device"];
	}

	bufferSize = outWidth * outHeight * BYTES_PER_PIXEL_OUT;

	destinationPoints[0] = Point2f{0,0};
	destinationPoints[1] = Point2f{(float)outWidth-1, 0};
	destinationPoints[2] = Point2f{0, (float)outHeight-1};
	destinationPoints[3] = Point2f{(float)outWidth-1, (float)outHeight-1};

	configFile["camera_matrix"] >> cameraMatrix;
	configFile["distortion_coefficients"] >> distortionCoefficients;

	Mat tempSourcePoints;
	configFile["corner_points"] >> tempSourcePoints;
	sourcePoints[0] = Point2f{tempSourcePoints.at<float>(0,0), tempSourcePoints.at<float>(0,1)};
	sourcePoints[1] = Point2f{tempSourcePoints.at<float>(1,0), tempSourcePoints.at<float>(1,1)};
	sourcePoints[2] = Point2f{tempSourcePoints.at<float>(2,0), tempSourcePoints.at<float>(2,1)};
	sourcePoints[3] = Point2f{tempSourcePoints.at<float>(3,0), tempSourcePoints.at<float>(3,1)};

	// cout << "0" << sourcePoints[0] << endl;
	// cout << "1" << sourcePoints[1] << endl;
	// cout << "2" << sourcePoints[2] << endl;
	// cout << "3" << sourcePoints[3] << endl;

	perspTransform = getPerspectiveTransform(sourcePoints, destinationPoints);

	// cout << "Dist:" << distortionCoefficients << endl;
	// cout << "CMat:" << cameraMatrix << endl;
	// cout << "Points:" << tempSourcePoints << endl;

	logLine("Successfully loaded config file.");
}

void openCamera()
{
	logLine("Opening camera " + cameraDevice);
    int deviceID = 1;             // 0 = open default camera
    int apiID = CAP_V4L2;      // 0 = autodetect default API
    // open selected camera using selected API
	// cap.open(deviceID + apiID);
	cap.open(cameraDevice);
	
    // check if we succeeded
    if (!cap.isOpened()) {
        throw("ERROR! Unable to open camera");
        return;
    }
}

void captureImage()
{
	logLine("Reading image.");
	// wait for a new frame from camera and store it into 'frame'
	cap.read(currentImage);
	logLine("Got image.");

	// check if we succeeded
	if (currentImage.empty()) {
		throw("ERROR! blank frame grabbed");
	}

	if(processedImage.empty())
	{
		currentImage.copyTo(processedImage);
	}

    return;
}

void initOutput()
{
	logLine("Initializing output device " + loopbackDevice);
	outputFD = open(loopbackDevice.c_str(), O_WRONLY);
	if (outputFD == -1)
	{
		throw("Error Opening Output device");
		return;
	}

	struct v4l2_capability vid_caps = {0};

	if(-1 == ioctl(outputFD, VIDIOC_QUERYCAP, &vid_caps))
	{
		throw("Query capture capabilites");
		return;
	}

	struct v4l2_format fmt = {0};
	fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
	
	if (-1 == ioctl(outputFD, VIDIOC_G_FMT, &fmt))
	{
		throw(std::string("Getting Pixel Format Output: ").append(strerror(errno)));
		return;
	}

	fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
	fmt.fmt.pix.width = outWidth;
	fmt.fmt.pix.height = outHeight;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB32;
	fmt.fmt.pix.sizeimage = bufferSize;
	fmt.fmt.pix.field = V4L2_FIELD_NONE;
	fmt.fmt.pix.bytesperline = outWidth * BYTES_PER_PIXEL_OUT;
	fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SMPTE170M;

	if (-1 == ioctl(outputFD, VIDIOC_S_FMT, &fmt))
	{
		throw("Setting Pixel Format Output");
		return;
	}

	outBuffer=(unsigned char *)malloc(sizeof(unsigned char) * bufferSize);
	memset(outBuffer, 0, bufferSize);
}

void outputImage()
{
	//Convert to RGBA

	if(rgbaImg.empty())
	{
		rgbaImg.create(bgraImg.rows, bgraImg.cols, CV_8UC4);
	}

	cvtColor(processedImage, rgbaImg, CV_BGR2RGBA);

	logLine("Writing to output");
	if(write(outputFD, rgbaImg.ptr(), bufferSize) <= 0)
	{
		throw("Write to output failed");
		return;
	}
}

void processImage()
{
	//Undistort the image by distortion coefficients
	// if(!cameraMatrix.empty())
	// {
	// 	undistort(currentImage, undistortedTemp, cameraMatrix, distortionCoefficients);
	// }
	// else
	// {
		// currentImage.copyTo(undistortedTemp);
	// }

	logLine("Processing image");

	//Perspective Transformation
	if(!perspTransform.empty())
	{
		// cout << perspTransform << endl;
		// warpPerspective(undistortedTemp, processedImage, perspTransform, Size{width, height});
		warpPerspective(currentImage, processedImage, perspTransform, Size{outWidth, outHeight});
	}
	else
	{
		processedImage = undistortedTemp;
	}
}

int main(int argc, char ** argv)
{
	logLine("OpenCV version: " + std::string(CV_VERSION));
	try
	{
		//Load config file
		loadConfig();

		//Open V4L2-capture device
		openCamera();

		//Open output V4L2-Device
		initOutput();

		//Perform the operation
		logLine("Starting processing loop.");
		while(true)
		{
			captureImage();	// The read inside this is blocking, so we don't need to sleep
			// imwrite("image.png", currentImage);

			processImage();
			// imwrite("processed.png", processedImage);

			// currentImage.copyTo(processedImage);
			outputImage();
		}
	}
	catch(std::exception const& e) {
		logLine(e.what());
	}
	catch(std::string & e)
	{
		logLine(e);
	}
	catch(const char * e)
	{
		logLine(std::string(e));
	}
	catch(...) {
		logLine("Exception occurred");
	}
	return 0;
}
