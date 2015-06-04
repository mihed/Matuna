# ATML
Automatically Tuned Machine Learning.

#Introduction
There's tons of stuff missing in the current alpha version. However, I try to follow my philosophy of releasing something that works without a ton of features to get additional experience for the next release. The current version was written in about 3-4 weeks (so, about 40 000 lines of code in this period!). To set up the entire cross-platform development system took me another week. There's a lot of duplicated code, no comments and a lot of things that are just partially implemented. The next alpha releases will not focus on any features but mostly tidying, refactoring and commenting in order to make this a usable library. 

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
