# TextureDefrag

![teaser](https://user-images.githubusercontent.com/13699526/116824395-32242c00-ab8a-11eb-9493-3c3b1af3d812.jpg)

A reference implementation for the paper [Texture Defragmentation for Photo-Reconstructed 3D Models](https://diglib.eg.org/handle/10.1111/cgf142615) by Andrea Maggiordomo, Paolo Cignoni and Marco Tarini.

### Abstract

We propose a method to improve an existing parametrization (UV-map layout) of a textured 3D model, targeted explicitly at alleviating typical defects afflicting models generated with automatic photo-reconstruction tools from real-world objects.
This class of 3D data is becoming increasingly important thanks to the growing popularity of reliable, ready-to-use photogrammetry software packages.
The resulting textured models are richly detailed, but their underlying parametrization typically falls short of many practical requirements, particularly exhibiting excessive fragmentation and consequent problems.
Producing a completely new UV-map, with standard parametrization techniques, and then resampling a new texture image, is often neither practical nor desirable for at least two reasons: first, these models have characteristics (such as inconsistencies, high resolution) that make them unfit for automatic or manual parametrization; second, the required resampling leads to unnecessary signal degradation because this process is unaware of the original texel densities.
In contrast, our method improves the existing UV-map instead of replacing it, balancing the reduction of the map fragmentation with signal degradation due to resampling, while also avoiding oversampling of the original signal.
The proposed approach is fully automatic and extensively tested on a large benchmark of photo-reconstructed models; quantitative evaluation evidences a drastic and consistent improvement of the mappings.

### Building and running

Dependencies:
 * QT5

Setup procedure for Ubuntu AWS EC2 instance with Nvidia GPU to build and run in headless environment. Uses virtual X server (Xvfb) with Nvidia hardware OpenGL rendering.

## 1. Instance Setup

**1.0 Install NVIDIA driver (GRID/vGPU)**

Follow AWS documentation: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-GRID-driver

**1.1 Install Build Tools and Qt5 Dependencies**
```bash
sudo apt-get update
sudo apt-get install -y build-essential qt5-qmake qtbase5-dev libgl1-mesa-dev
```

**1.2 Install Virtual Framebuffer (Xvfb)**
```bash
sudo apt-get install -y xvfb
```

**1.3 Set User Permissions for GPU Access**
```bash
sudo usermod -a -G video,render ubuntu
```

**IMPORTANT:** Log out and reconnect SSH session or reboot:
```bash
# EITHER log out and reconnect
exit

# OR reboot
sudo reboot
```

## 2. Building the Application

```bash
# Navigate to project directory, then:
rm -rf build
mkdir build && cd build
qmake ../texture-defrag/texture-defrag.pro -spec linux-g++
make -j32
```

## 3. Running the Application

Set `__GLX_VENDOR_LIBRARY_NAME=nvidia` to force Nvidia hardware OpenGL rendering instead of software `llvmpipe` renderer:

**Interactive**
```bash
__GLX_VENDOR_LIBRARY_NAME=nvidia \
xvfb-run --auto-servernum ./texture-defrag \
~/consor/merlin_textured.obj -o ~/ts/processed.obj -l 1 -g 99999.0 -r 4 -c 5 -p 80
```

Verify output shows: `[GL] Vendor: NVIDIA Corporation`

**Background**
```bash
nohup env __GLX_VENDOR_LIBRARY_NAME=nvidia \
xvfb-run --auto-servernum ./texture-defrag \
~/consor/merlin_textured.obj \
-o ~/ts/processed.obj \
-l 1 -g 99999.0 -r 1 -c 5 -p 80 \
> ~/ts/processing.log 2>&1 &
```

```
Usage: ./texture-defrag MESHFILE [-mbdgutao]

MESHFILE specifies the input mesh file (supported formats are obj, ply and fbx)

-m  <val>      Matching error tolerance when attempting merge operations. (default: 2)
-b  <val>      Maximum tolerance on the seam-length to chart-perimeter ratio when attempting merge operations. Range is [0,1]. (default: 0.2)
-d  <val>      Local ARAP distortion tolerance when performing the local UV optimization. (default: 0.5)
-g  <val>      Global ARAP distortion tolerance when performing the local UV optimization. (default: 0.025)
-u  <val>      UV border reduction target in percentage relative to the input. Range is [0,1]. (default: 0)
-a  <val>      Alpha parameter to control the UV optimization area size. (default: 5)
-t  <val>      Time-limit for the atlas clustering (in seconds). (default: 0)
-o  <val>      Output mesh file. Supported formats are obj and ply. (default: out_MESHFILE)
-l  <val>      Logging level. 0 for minimal verbosity, 1 for verbose output, 2 for debug output. (default: 0)

```

### Citation

```
@article {maggiordomo2021defragmentation,
  journal = {Computer Graphics Forum},
  title = {{Texture Defragmentation for Photo-Reconstructed 3D Models}},
  author = {Maggiordomo, Andrea and Cignoni, Paolo and Tarini, Marco},
  year = {2021},
  publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
  ISSN = {1467-8659},
  DOI = {10.1111/cgf.142615}
}
```
