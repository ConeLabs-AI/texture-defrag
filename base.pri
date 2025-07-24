VCGPATH = $$PWD/vcglib

CONFIG += console
CONFIG += c++11
CONFIG -= app_bundle

#### QT STUFF ##################################################################

TEMPLATE = app
QT = core gui svg

##### INCLUDE PATH #############################################################

INCLUDEPATH += $$PWD/src $$VCGPATH $$VCGPATH/eigenlib

#### PLATFORM SPECIFIC #########################################################

unix|mingw-g++ {
    # For GCC and Clang on Unix-like systems (including MinGW-g++)
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LDFLAGS += -fopenmp
}

win32-msvc* {
    # For Microsoft Visual C++ compiler on Windows
    QMAKE_CXXFLAGS += /openmp
    # MSVC does not typically require a separate linker flag for OpenMP
}


unix {
  LIBS += -lGLU
}

win32 {
  LIBS += -lopengl32
  DEFINES += NOMINMAX
}
