# ATML
Automatically Tuned Machine Learning.


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

TODO
