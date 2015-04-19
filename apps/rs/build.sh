mkdir jni/prebuilt/
cp ${ANDROID_HOME:-/Volumes/android/trunk}/out/target/product/shamu/system/lib/libRS.so jni/prebuilt/libRS.so

pushd ../..
make -j8 || exit 
popd
# g++ -std=c++11 -I ../../include/ copy.cpp -L ../../bin/ -lHalide -o copy -g
# LD_LIBRARY_PATH=../../bin DYLD_LIBRARY_PATH=../../bin HL_DEBUG_CODEGEN=4 WITH_RS=1 HL_TARGET=arm-32-android-armv7s-rs-debug ./copy 2> copy-rs-out
g++ -std=c++11 -I ../../include/ blur.cpp -L ../../bin/ -lHalide -o blur -g
LD_LIBRARY_PATH=../../bin DYLD_LIBRARY_PATH=../../bin HL_DEBUG_CODEGEN=4 WITH_RS=1 HL_TARGET=arm-32-android-armv7s-rs-debug ./blur 2> blur-rs-out
