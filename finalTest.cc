#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


// Filter implemantations


void unsharpenMask(Mat in);

void laplacianFilter(Mat inputImage, double& intensity) {
  Mat laplacian;
  Laplacian(inputImage, laplacian, inputImage.depth(), 3);

  imshow("laplacian", laplacian);
  convertScaleAbs(laplacian, laplacian);

  int yesOrNo;
  cout << " Is the image noisy (type 1 to apply gussian blur to laplacain "
          "mask, 0 for no)\n";
  cin >> yesOrNo;
  if(yesOrNo == 1) {
    GaussianBlur(laplacian, laplacian, Size(3, 3), 0, 0);
  }

  laplacian *= intensity;
  imshow("laplacian", laplacian);
  Mat sharp;
  sharp = inputImage + (-1 * laplacian);
  imshow("sharp", sharp);
  imwrite("laplacianSharp.png", sharp);
  imwrite("laplacianMask.png", laplacian);
  waitKey(0);
}
void unsharpenMask(Mat in, double &intensity) {
  Mat input = in.clone();
  
  Mat blurred;

  GaussianBlur(input, blurred, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);

  imshow("blurred.jpeg", blurred);
  
  Mat unsharpMask;
  cv::subtract(input, blurred, unsharpMask);
  unsharpMask = unsharpMask * intensity;
  imshow("unsharp", unsharpMask);
  Mat sharp;
  GaussianBlur(unsharpMask, unsharpMask, Size(3, 3), 1.4);
  addWeighted(input, 1, unsharpMask, 1, 0, sharp);

  imshow("sharp.jpeg", sharp);
  waitKey(0);
  
  imwrite("unsharpenMask.png", unsharpMask);
  imwrite("unsharpenMask-sharpened.png", sharp);
  return;
}

void Roberts(Mat input, double &intensity) {
  imshow("input", input);
  
  Mat Roberts_x = (Mat_<double>(2, 2) << -1, 0, 0, 1);
  Mat Roberts_y = (Mat_<double>(2, 2) << 0, -1, 1, 0);

  Mat G_x;
 
  filter2D(input, G_x, input.depth(), Roberts_x);

  Mat G_y;
  
  filter2D(input, G_y, input.depth(), Roberts_y);

  convertScaleAbs(G_x, G_x);
  convertScaleAbs(G_y, G_y);

  G_x.convertTo(G_x, CV_32F);
  G_y.convertTo(G_y, CV_32F);

  Mat G_xy;

  magnitude(G_x, G_y, G_xy);
  
  convertScaleAbs(G_xy, G_xy);
  imshow("mag", G_xy);
    G_xy *= intensity;
    Mat output;
    add(input, G_xy, output);

    imshow("output", output);
    imwrite("RobertsSharpen.png", output);
    imwrite("RobertsMask.png", G_xy);

    waitKey(0);
}

void sobel(Mat input, double &intensity) {
  imshow("in", input);
  
  Mat Sobel_x = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
  Mat Sobel_y = (Mat_<double>(3, 3) << -1,0,1,-2,0,2,-1,0,1);

  Mat G_x;
  filter2D(input, G_x, input.depth(), Sobel_x);
  imshow("G_x", G_x);

  Mat G_y;
  filter2D(input, G_y, input.depth(), Sobel_y);
  imshow("G_y", G_y);

  G_x.convertTo(G_x, CV_32F);
  G_y.convertTo(G_y, CV_32F);
  Mat G_xy;
  magnitude(G_x, G_y, G_xy);
  convertScaleAbs(G_xy, G_xy);

  G_xy *= intensity;
  imshow("mag", G_xy);

  Mat output;
  add(input, G_xy, output);
  imshow("output", output);
  imwrite("Sobel.png", output);
  imwrite("SobelMask.png", G_xy);

  waitKey(0);
}



void perwitt(Mat input, double &intensity) {
  imshow("in", input);
  
  Mat Sobel_x = (Mat_<double>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
  Mat Sobel_y = (Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
 
  Mat G_x;
  filter2D(input, G_x, input.depth(), Sobel_x);
  imshow("G_x", G_x);

  Mat G_y;
  filter2D(input, G_y, input.depth(), Sobel_y);
  imshow("G_y", G_y);

  G_x.convertTo(G_x, CV_32F);
  G_y.convertTo(G_y, CV_32F);

  Mat G_xy;
  magnitude(G_x, G_y, G_xy);
  convertScaleAbs(G_xy, G_xy);
  G_xy *= intensity;
  imshow("G_xy", G_xy);

  Mat output;
  add(input, G_xy, output);

  imshow("output", output);
  imwrite("PrewittSharpen.png", output);
  imwrite("PrewittMask.png", G_xy);

  waitKey(0);
}

void equalizeHistogramGRAY( Mat input) {
  cvtColor(input, input, COLOR_BGR2GRAY);
  Mat dst;
  
  equalizeHist(input, dst);
  imshow("Source image", input);
  imshow("Equalized Image", dst);
  imwrite("EqualizedImage.png", dst);
  waitKey();
  return;
}

void equalizeHistogramCOLOR( Mat input) {
  Mat equalizedImage;
  cvtColor(input, equalizedImage, COLOR_BGR2YCrCb);

  vector<Mat> channels;
  split(equalizedImage, channels);

  equalizeHist(channels[0], channels[0]);

  merge(channels, equalizedImage);

  cvtColor(equalizedImage, equalizedImage, COLOR_YCrCb2BGR);

  imshow("equalized", equalizedImage);
  imwrite("ColorEqualized.png", equalizedImage);

  waitKey(0);  
}

int main(int argc, char** argv) {
  int input;
  Mat inputImage;
  String originalImage = "Original";
  inputImage = imread(argv[1], IMREAD_COLOR);
  string a, b;
  cout << "Choose an image processing technique: \n\n";
  cout << "1: show original image\n";
  cout << "2: Sharpen Image Using laplace filter\n";
  cout << "3: Sharpen using unsharpen mask\n";
  cout << "4: Roberts\n";
  cout << "5: Sobel\n";
  cout << "6: Perwitt\n";
  cout << "7: Equalize histogram for gray image\n";
  cout << "8: Equalize histogram for colored image\n";
  

  double intensity;
  int yesOrNo;
  cin >> input;

  switch (input) {
    case 1:
      namedWindow(originalImage);
      imshow(originalImage, inputImage);
      waitKey(0);
      destroyWindow(originalImage);
      break;
    case 2:
      cout << "please specify the intesity of the sharpeing.\nUse 1 as default before increasing it\n";
      cin >> intensity;
     
      laplacianFilter(inputImage,intensity);
      break;
    case 3:

     
      cout << "please specify the intensity of the sharpening\n";
      cin >> intensity;
      unsharpenMask(inputImage, intensity);

      cout << "Change intensity?\n Use value 1 as yes and value 0 as no";
      cin >> yesOrNo;
      while(yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        unsharpenMask(inputImage, intensity);
        cout << "change again?\nUse value 1 as yes an value 0 as no";
        cin >> yesOrNo;
      }
      break;

    case 4:
      
      cout << "please specify the intesity of the sharpeing\nUse 1 as default before increasing it\n";
      cin >> intensity;
      Roberts(inputImage, intensity);

      cout << "Chanage intensity?\nUse value 1 as yes and value 0 as no";
      cin >> yesOrNo;
      while (yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        Roberts(inputImage, intensity);
        cout << "change again?\nUse value 1 as yes and value 0 as no";
        cin >> yesOrNo;
      }
      break;
    case 5:

      cout << "please specify the intesity of the sharpeing\nUse 1 as default before increasing it\n";
      cin >> intensity;
      sobel(inputImage, intensity);

      cout << "Chanage intensity?\nUse value 1 as yes and value 0 as no";
      cin >> yesOrNo;
      while (yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        sobel(inputImage, intensity);
        cout << "change again?\nUse value 1 as yes and value 0 as no";
        cin >> yesOrNo;
      }

      break;
    case 6:
      cout << "please specify the intesity of the sharpeing\nUse 1 as default "
              "before increasing it\n";
      cin >> intensity;
      perwitt(inputImage, intensity);

      cout << "Chnage intensity?\nUse value 1 as yes and value 0 as no";
      cin >> yesOrNo;
      while (yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        perwitt(inputImage, intensity);
        cout << "change again?\nUse value 1 as yes and value 0 as no";
        cin >> yesOrNo;
      }
      break;
  }
}