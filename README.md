# 🔄 Parallel Computing and Multimedia Processing

A C++ project repository dedicated to CPU-based multithreaded computing for image and video processing tasks. Developed for academic labs and research in multimedia processing and parallel computing, this repository supports OpenCV-based implementations and structured lab submissions.

---


## 🚀 Lab 1: Video Pixel Intensity Highlighting (Variant 2)

### 🎯 Objective

Process a video frame-by-frame to:
- Convert each frame to grayscale
- Identify pixels with intensity `< 64`
- Highlight them in red on the original frame
- Use multithreading to boost performance
- Save the result as a video

### 🛠️ Built With

- **C++17**
- **OpenCV 4**
- **POSIX Threads (`<thread>`)**
- **CMake** (Cross-platform build automation)
- Tested on **macOS + VS Code**

---

## 🧪 How to Build and Run (macOS)

### 🔧 Step 1: Install OpenCV
```bash
brew install opencv
export PKG_CONFIG_PATH="/opt/homebrew/opt/opencv/lib/pkgconfig"

##**🔧 Step 2: Compile with CMake**

mkdir build && cd build
cmake ..
make
./lab1

