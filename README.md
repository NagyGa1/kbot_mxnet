# kbot_mxnet

MXNetwork java class wrapping implemented with MXNet via JNI / C++.

## Compile MXNet
```shell script
cd ~/git
git clone https://github.com/apache/incubator-mxnet mxnet
cd mxnet
git checkout 1.6.0
git submodule update --init --recursive
```

If no python but there is python3 (default Ubuntu):
```shell script
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 3
```

Follow [https://mxnet.apache.org/versions/1.6/get_started/build_from_source](https://mxnet.apache.org/versions/1.6/get_started/build_from_source)
```shell script
sudo apt-get install -y build-essential git ninja-build ccache libopenblas-dev libopencv-dev cmake
```

```shell script
mkdir build
cd build
cmake -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_MKLDNN=1 -DUSE_CPP_PACKAGE=1 -DCMAKE_BUILD_TYPE=Release ..
make -j 8
sudo make install
sudo ldconfig
```

## Compile kbot_mxnet jni
```shell script
cd kbot_mxnet/jni
mkdir cmake-build-release
cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
sudo make install
```

Places itself to `/usr/lib/jni` where java can pick it up without any other magic.

Uninstall kbot_native (not provided for mxnet)
```shell script
sudo make uninstall
```

### Compile java package and place to maven local repo

```shell script
cd kbot_mxnet
./gradlew clean test publishToMavenLocal --info
```