# Matuna
Machine learning with Automatically Tuned Algorithms.

The alpha versions runs OK on CPU but slow on GPU since no optimizations are made on these versions. (work in progress...)

#Introduction
This library was mainly an effort for testing new state-of-the-art machine learning algorithms. Since there's already plenty of libraries out there, I wanted to put some extra focus on applications and optimizations.
The library is cross platform, and I've compiled it so far for Linux, OS X, iOS, Windows and Android.
Compilation of OpenCL kernels are automatically supported for almost all OpenCL platform (except some embedded profiles), which leaves a possibility of adjusting the kernel code before compilation and execution.
This is what Automatically Tuned refers to. The device information such as constant cache size, local cache size, preferred vectorization etc... are read out and the corresponding kernel is adapted.
The latest alpha version does not contain a lot of functionality, and will not do so before the code has been cleaned up (40 000 lines of code in 3 weeks, does indeed has its side-effects!).
Depending on my professional engagements, I will try to push out new implementations as fast as possible using the latest research in the Machine Learning community. 
In the ConvNet, I've tried to avoid as much as possible of memory copies in order to reduce memory usage and kernel executions. Instead, every layer is reading directly from the previous layer's output memory. 
Check out the ConvNet notes for more information about the memory layout or if you just want to get all the CNN equations OOTB.

##Development Environment (Sketch)
###Windows
- CUDA
- Cmake 3.2 (Minimum version)
- Eclipse Luna
- Visual Studio 2013
- TDM-GCC-64

If you are using this, you should be able to build the project out of the box. I prefer developing with Eclipse since it makes it easier to work with custom source locations. I use VS for compilation to make sure the code is compatible with Windows.

###Ubuntu, Android
- CUDA
- Cmake 3.2 (Minimum version)
- Android Studio (or standalone Android SDK)
- Eclipse Luna with ADT
- Android NDK

However, in order to build for Android devices you must check the following in your environment variables:
ANDROID_CMAKE=<path>/ATML/thirdparty/share/cmake
ANDROID_NDK=<path-to-your-NDK-folder>

Furthermore, you need to create a standalone toolchain using the NDK and place it under /op/android-toolchain. How you create this toolchain is described in your NDK folder under /docs. When this is done, you simply fire up cmake and indicate the android.toolchain.cmake file after checking "Specify toolchain for cross-compiling". After building you simply type make in the build folder and you have android-compiled binaries for you android applications.

####Remark
I included the android.toolchain.cmake to make life easier. However, you could simply clone https://github.com/taka-no-me/android-cmake to have the latest version - which I highly recommend you to do.

###Mac OS, iOS
- Cmake 3.2 (Minimum version)
- Xcode

Since Cmake automatically generates the Xcode project for you, you may simply change the platform to either MAC OS X or iOS. The library is developed using Xcode 6.3 on MAC OS X 10.10. 
