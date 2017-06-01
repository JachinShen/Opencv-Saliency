#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//傅里叶正变换
void fft2(Mat &src, Mat &dst);

int main()
{
    string imagePath = "RoboMaster.png";

    //////////////////////////////////////////////////////////////////////////
    //显著性计算
    //参考论文：Saliency Detection: A Spectral Residual Approach
    
    //amplitude和phase分别是图像的振幅谱和相位谱
    Mat src, ImageRe, ImageIm, Fourier, Inverse, LogAmplitude,  Sine, Cosine;
    Mat Saliency, Residual;
    Mat tmp1, tmp2, tmp3;
    //double minNum = 0, maxNum = 0, scale, shift;
    //int i, j,     nRow, nCol;
    
    //加载源图像，第二个参数为0表示将输入的图片转为单通道，大于0为三通道
    src = imread(imagePath.c_str(),0);
    //注意Fourier是一个两通道的图像，一个通道为实部，一个为虚部
    Fourier.create(src.rows, src.cols, CV_64FC2);  
    Inverse.create(src.rows, src.cols, CV_64FC2);  
    //频谱的实部
    ImageRe.create(src.rows, src.cols, CV_64FC1);  
    //频谱的虚部
    ImageIm.create(src.rows, src.cols, CV_64FC1);  
    //log振幅谱
    LogAmplitude.create(src.rows, src.cols, CV_64FC1);  
    //正弦谱
    Sine.create(src.rows, src.cols, CV_64FC1);  
    //余弦谱
    Cosine.create(src.rows, src.cols, CV_64FC1);  

    //频谱冗余（spectral residual）
    Residual.create(src.rows, src.cols, CV_64FC1);  
    //特征map(Saliency map)
    Saliency = src.clone(); 

    //临时的空间
    tmp1.create(src.rows, src.cols, CV_64FC1);  
    tmp2.create(src.rows, src.cols, CV_64FC1);  
    tmp3.create(src.rows, src.cols, CV_64FC1);  

   // nRow = src.rows;
    //nCol = src.cols;

    //归一化一下
   // scale = 1.0/255.0;
	//
	normalize(src, tmp1, 1.0, 0.0, NORM_MINMAX);
	


    //傅里叶变换，得到的Fourier有两个通道，一个是实部，一个是虚部
    fft2(tmp1, Fourier);

    //将傅里叶谱的实部和虚部存放到对应的图像中去。
	vector<Mat> channels;
    split(Fourier, channels); 
	ImageRe = channels.at(0);
	ImageIm = channels.at(1);
	

    //计算傅里叶振幅谱，实部和虚部平方和再开方，得到振幅谱存到tmp3中
    pow( ImageRe, 2.0, tmp1);
    pow( ImageIm, 2.0, tmp2);
    add( tmp1, tmp2, tmp3);
    pow( tmp3, 0.5, tmp3 );
	

    //计算正弦谱和余弦谱和自然对数谱
    log( tmp3, LogAmplitude );
    divide(ImageIm, tmp3, Sine);
    divide(ImageRe, tmp3, Cosine);
	

    //对LogAmplitude做3*3均值滤波
    blur(LogAmplitude, tmp3, Size(3, 3));

    //计算出剩余普
    subtract(LogAmplitude, tmp3, Residual);

       exp(Residual, Residual);
    multiply(Residual, Cosine, tmp1);
    multiply(Residual, Sine, tmp2);
	
    //将剩余普Residual作为实部，相位谱Phase作为虚部
	channels.at(0) = tmp1;
	channels.at(1) = tmp2;
    merge(channels,  Fourier);
    
    //实现傅里叶逆变换
    dft(Fourier, Inverse, DFT_INVERSE);

    split(Inverse , channels);
	tmp1 = channels.at(0);
	tmp2 = channels.at(1);
	//imshow("debug1", Residual);
	//imshow("debug2", tmp1);
	//imshow("debug3", tmp2);
	//waitKey();

    //求出对应的实部虚部平方和
    pow(tmp1, 2, tmp1);
    pow(tmp2, 2, tmp2);
    add(tmp1, tmp2, tmp3);

    //高斯滤波
    //GaussianBlur(tmp3, tmp3, Size(3, 3), 0, 0);

    //minMaxLoc(tmp3, &minNum, &maxNum, NULL, NULL);
    //scale = 255/(maxNum - minNum);
    //shift = -minNum * scale;

    //将shift加在ImageRe各元素按比例缩放的结果上，存储为ImageDst
    blur(tmp3, Saliency, Size(3, 3));

    namedWindow("Saliency", 1);
    imshow("Saliency",Saliency);

    waitKey(0);

    //释放图像

    return 0;
}

/**************************************************************************
//src IPL_DEPTH_8U
//dst IPL_DEPTH_64F
**************************************************************************/
//傅里叶正变换
void fft2(Mat &I, Mat &dst)
{   //实部、虚部
    Mat image_Re , image_Im , Fourier ;
    //   int i, j;
	//image_Re.create(src.rows, src.cols, CV_64FC1);
    //Imaginary part
	//image_Im.create(src.rows, src.cols, CV_64FC1);
    //2 channels (image_Re, image_Im)

    /************************************************************************/
    /*
    void cvConvertScale( const CvArr* src, CvArr* dst, double scale=1, double shift=0 );
    src
    原数组. 
    dst
    输出数组 
    scale
    比例因子. 
    shift
    原数组元素按比例缩放后添加的值。 

    函数 cvConvertScale 有多个不同的目的因此就有多个意义，
    函数按比例从一个数组中拷贝元素到另一个元素这种操作是最先执行的，
    或者任意的类型转换，正如下面的操作：

    dst(I)=src(I)*scale + (shift,shift,...) 

    多通道的数组对各个地区通道是独立处理的。
    */
    /************************************************************************/
	Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<double>(padded), Mat::zeros(padded.size(), CV_64FC1)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros    //实部的值初始设为源图像，虚部的值初始设为0
    // Real part conversion from u8 to 64f (double)
	//image_Re = src.clone();
    // Imaginary part (zeros)
	//image_Im = Mat::zeros(src.rows, src.cols, CV_64FC1);
    // Join real and imaginary parts and stock them in Fourier image
	//
	//vector<Mat>channels;
	//split(Fourier, channels);
	//channels.at(0) = image_Re.clone();
	//
	//channels.at(1) = image_Im.clone();
	//imshow("Debug", complexI);
    // Application of the forward Fourier transform
    dft(complexI, dst, DFT_COMPLEX_OUTPUT);
}
