VCGPATH = $$PWD/vcglib

CONFIG += console
CONFIG += c++14
CONFIG += release
CONFIG -= app_bundle

#### QT STUFF ##################################################################

TEMPLATE = app
QT = core gui svg

##### INCLUDE PATH #############################################################

INCLUDEPATH += $$PWD/src $$VCGPATH $$VCGPATH/eigenlib

DEFINES += EIGEN_DONT_ALIGN_STATICALLY
DEFINES += EIGEN_MAX_STATIC_ALIGN_BYTES=0

#### PLATFORM SPECIFIC #########################################################

unix|mingw-g++ {
    # For GCC and Clang on Unix-like systems (including MinGW-g++)
    !macx {
        QMAKE_CXXFLAGS += -fopenmp
        LIBS += -fopenmp
    }
    QMAKE_CXXFLAGS += -march=native
    QMAKE_CXXFLAGS_RELEASE -= -O
    QMAKE_CXXFLAGS_RELEASE -= -O1
    QMAKE_CXXFLAGS_RELEASE -= -O2
    QMAKE_CXXFLAGS_RELEASE += -O3
}

win32-msvc* {
    # For Microsoft Visual C++ compiler on Windows
    QMAKE_CXXFLAGS += /openmp
    QMAKE_CXXFLAGS += /arch:AVX2
    # MSVC does not typically require a separate linker flag for OpenMP
}


unix {
  LIBS += -lGLU
}

win32 {
  LIBS += -lopengl32
  DEFINES += NOMINMAX
}