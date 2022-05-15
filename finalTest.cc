#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


// Filter implemantations


void unsharpenMask(Mat in);

void laplacianFilter(Mat inputImage) {
  Mat laplcain;
  Laplacian(inputImage, laplcain, inputImage.depth(), 3);
 
  imshow("laplcian", laplcain);
  

  convertScaleAbs(laplcain, laplcain);
  
  Mat in = inputImage.clone();
  
  Mat laplacian;
  Laplacian(in, laplacian, in.depth(), 3);


  imshow("laplacian", laplacian);
  convertScaleAbs(laplacian, laplacian);

  GaussianBlur(laplacian, laplacian, Size(3, 3), 1.4, 0);

  Mat sharp;
  sharp = in + (-1 * laplacian);
  imshow("sharp", sharp);
  imwrite("laplcianSharp.png", sharp);
  waitKey(0);

}
void unsharpenMask(Mat in, double &intensity) {
  Mat input = in.clone();
  
  Mat blurred;

  cv::GaussianBlur(input, blurred, cv::Size(3, 3), 0, 0, BORDER_DEFAULT);

  cv::imshow("blurred.jpeg", blurred);
  

  Mat unsharpMask;
  cv::subtract(input, blurred, unsharpMask);
  unsharpMask = unsharpMask * intensity;
  imshow("unsharp", unsharpMask);
  Mat sharp;
  GaussianBlur(unsharpMask, unsharpMask, Size(3, 3), 1.4);
  addWeighted(input, 1, unsharpMask, 1, 0, sharp);

  imshow("sharp.jpeg", sharp);
  waitKey(0);
  destroyWindow("sharp.jpeg");
  return;
}

void Roberts(Mat input, double &intensity) {
  imshow("input", input);
  Mat RGB[3];
  split(input, RGB);
  Mat Roberts_x = (Mat_<double>(2, 2) << -1, 0, 0, 1);
  Mat Roberts_y = (Mat_<double>(2, 2) << 0, -1, 1, 0);

  Mat G_xRGB[3];
 
  
  filter2D(RGB[0], G_xRGB[0], input.depth(), Roberts_x);
  filter2D(RGB[1], G_xRGB[1], input.depth(), Roberts_x);
  filter2D(RGB[2], G_xRGB[2], input.depth(), Roberts_x);

  Mat G_yRGB[3];
  

  filter2D(RGB[0], G_yRGB[0], input.depth(), Roberts_y);
  filter2D(RGB[1], G_yRGB[1], input.depth(), Roberts_y);
  filter2D(RGB[2], G_yRGB[2], input.depth(), Roberts_y);

  Mat G_xTotal;
  Mat G_yTotal;
  merge(G_xRGB, 3, G_xTotal);
  merge(G_yRGB, 3, G_yTotal);

  convertScaleAbs(G_xTotal, G_xTotal);
  convertScaleAbs(G_yTotal, G_yTotal);

  G_xTotal.convertTo(G_xTotal, CV_32F);
  G_yTotal.convertTo(G_yTotal, CV_32F);

  
  Mat G_xy;

  
  magnitude(G_xTotal, G_yTotal, G_xy);
  

  convertScaleAbs(G_xy, G_xy);
  imshow("mag", G_xy);
    G_xy *= intensity;
    Mat output;
    add(input, G_xy, output);

    imshow("output", output);
    imwrite("RobertsSharpen.png", output);

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
  imwrite("PerwtitSharpen.png", output);

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
  Mat hist_equalized_image;
  cvtColor(input, hist_equalized_image, COLOR_BGR2YCrCb);

  vector<Mat> vec_channels;
  split(hist_equalized_image, vec_channels);

  
  equalizeHist(vec_channels[0], vec_channels[0]);

  
  merge(vec_channels, hist_equalized_image);

  
  cvtColor(hist_equalized_image, hist_equalized_image, COLOR_YCrCb2BGR);

  
  String windowNameOfOriginalImage = "Original Image";
  String windowNameOfHistogramEqualized = "Histogram Equalized Color Image";

  
  namedWindow(windowNameOfOriginalImage, WINDOW_NORMAL);
  namedWindow(windowNameOfHistogramEqualized, WINDOW_NORMAL);

  
  imshow(windowNameOfOriginalImage, input);
  imshow(windowNameOfHistogramEqualized, hist_equalized_image);
  imwrite("ColorEquilized.png", hist_equalized_image);

  waitKey(0);  
}

int main() {
  int input;
  Mat inputImage;
  String originalImage = "Original";
  inputImage = imread("ColorEquilized.png", IMREAD_COLOR);

  cout << "Choose a image processing technique: \n\n";
  cout << "1: show origial image\n";
  cout << "2: Sharpen Image Using laplace filter\n";
  cout << "3: Sharpen using unsharpen mask\n";
  cout << "4: Roberts\n";
  cout << "5: Sobel\n";
  cout << "6: Perwitt\n";
  cout << "7: Equilize Historam for gray image\n";
  cout << "8: Equailize histgram for colored image\n";

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

      laplacianFilter(inputImage);
      break;
    case 3:

     
      cout << "please specify the intesity of the sharpeing\n";
      cin >> intensity;
      unsharpenMask(inputImage, intensity);

      cout << "Chnage intensity?\n";
      cin >> yesOrNo;
      while(yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        unsharpenMask(inputImage, intensity);
        cout << "change again?\n";
        cin >> yesOrNo;
      }
      break;

    case 4:
      
      cout << "please specify the intesity of the sharpeing\n";
      cin >> intensity;
      Roberts(inputImage, intensity);

      cout << "Chnage intensity?\n";
      cin >> yesOrNo;
      while (yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        Roberts(inputImage, intensity);
        cout << "change again?\n";
        cin >> yesOrNo;
      }
      break;
    case 5:

      cout << "please specify the intesity of the sharpeing\n";
      cin >> intensity;
      sobel(inputImage, intensity);

      cout << "Chnage intensity?\n";
      cin >> yesOrNo;
      while (yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        sobel(inputImage, intensity);
        cout << "change again?\n";
        cin >> yesOrNo;
      }

      break;
    case 6:
      cout << "please specify the intesity of the sharpeing\n";
      cin >> intensity;
      perwitt(inputImage, intensity);

      cout << "Chnage intensity?\n";
      cin >> yesOrNo;
      while (yesOrNo == 1) {
        cout << "please state new intensity\n";
        cin >> intensity;
        perwitt(inputImage, intensity);
        cout << "change again?\n";
        cin >> yesOrNo;
      }
      break;

      case 7:
        equalizeHistogramGRAY(inputImage);
        break;
      case 8:
        equalizeHistogramCOLOR(inputImage);
        break;
  }
}