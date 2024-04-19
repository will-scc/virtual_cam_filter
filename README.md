### Virtual Cam Filter

A basic Python 3 script which uses `OpenCV` and `PyVirtualCam` to take a USB webcam input, detect and track faces, apply a filter (default a blur) and then create a virtual webcam which can be used in any other software.

Requires `opencv-python` and `pyvirtualcam`. Additionally, in order to use CUDA (i.e. shift workload from CPU to GPU) you must use `opencv-python` built against CUDA and cuDNN. Recommend picking a relevant wheels from here:

`https://github.com/cudawarped/opencv-python-cuda-wheels/releases`
