# Get the superresolution code running

In this text several remarks are given for anyone who wants to use the superresolution code.

## Installing CUDA
1. Find out what your GPU's compute capability is. Depending on that install the proper CUDA toolkit. E.g. for CUDA 9.0 CC 3.0 is required, for CUDA 8.0  CC: 2.1, for CUDA 6.5  CC: 1.1 and for CUDA 6.0  CC: 1.0
2. Add all dependencies to the PATH. Try the command "$ nvcc --version" to find out whether the CUDA compiler is working/is found.
In .bashrc: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/lib/nvidia-387"
Adjust everything to your computer!
3. Never mess things up with the drivers! Don't install newer drivers. It can horribly mess up things. On Ubuntu you should get the proper drivers included in the .deb-installation method of CUDA. Just follow the installation guide of NVIDIA.
4. If you messed something up with the drivers install from scratch with "sudo apt-get --reinstall install nvidia-current"

### Some frequent errors
- "warning: libnvidia-fatbinaryloader.so.384.111 not found" or similar. Leading to ".../libcuda.so: undefined reference to...". --> Solution: "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/lib/nvidia-387" or "Copy the missing lib (fatbinaryloader) in the same folder as libcuba.so. or make a symbolic link[ e.g. sudo ln -s /usr/lib/nvidia-387/libnvidia-fatbinaryloader.so
.387.26 /usr/lib/x86_64-linux-gnu/libnvidia-fatbinaryloader.so.387.26
]"

## Installing OpenCV
1. This code seems to work only with OpenCV 2.4
2. Download OpenCV 2.4 and build it according to the description on OpenCV website.
3. If during compilation you have a opencv_dep_cudart library missing put a symbolic link : "sudo ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libopencv_dep_cudart.so"

## CUDA theory

- https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
