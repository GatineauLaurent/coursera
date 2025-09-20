# Grayscale 24b/32b image conversion using NVIDIA NPP with CUDA

## Overview

This project is submission for the assignement of the coursera's module CUDA at Scale for the Enterprise (Independent Project).

The program will perform grayscale conversion on list of 24b or 32b images. Output images will be in same file format as input image, using same encoding format (24b or 32b per pixel).

Why not converting directly to 8b grayscale image?
- goal of assignement is to use NVIDIA NPP functions...
- professionnal image manipulation software work on separate color channels...
- fun...

## Requirements

- Linux, tested only in Cousrera lab enviromnent
- FreeImage
- NVIDIA NPP with CUDA

## How to build and run

- Clone and build the project:
```
$ git clone https://github.com/GatineauLaurent/coursera.git
$ cd coursera
$ make
```

- Run the project using provided images:
```
$ make run
```

- Run the project by yourself:
```
$ bin/grayScaleConverterNPP --help

bin/grayScaleConverterNPP [--help|-h] --input [file1|file2|...] --output_dir output
   --input: list of input files
   --output_dir: output directory (it must exist!)

Convert all input files to grayscale and put results in output directory.
!!!Files in output directory will be overwritten!!!
```

## Code Explanation

### Code Organization

- `bin/` This directory holds the binary code (must be build).

- `data/` This directory contains some example data.

- `output/` This directory will be created after `make run` command, and will contain output example data.

- `src/` This directory contains source code of the project.

- `README.md` This README file.

- `LICENSE` License of this project.

- `Makefile` Top `Makefile` used for:
  - Compilation with `make`.
  - Clean-up with `make clean`.
  - Run the project with `make run`.
  - Compilation flags and include/library locations are declared in this file.

### Source code organization

- `src/Makefile` should be called by the top `Makefile`.

- `src/main.cc` Contains the `main` function which is responsible of:
  - Parsing command line arguments.
  - Checking command line arguments correctness and display usage in case of error.
  - Calling `process` function for each input file.

- `src/process.h` Header file containing the prototype of the `process` function.

- `src/process.cc` Contains the function `process` function responsible of:
  - Checking input file is a supported image file (`FreeImage_GetFileType`, `FreeImage_GetImageType`, `FreeImage_GetBPP`).
  - Managing host memory for the input and output file (`FreeImage_Allocate`, `FreeImage_Unload`).
  - Reading input file and writing input file using FreeImage library (`FreeImage_Load`, `FreeImage_Save`).
  - Managing device memory for the input file, and NPP plans (`cudaMalloc`, `nppiMalloc_8u_C1`, `nppiFree`, `cudaFree`).
  - Copying data fro host to device and device to host (`cudaMemcpy`)
  - Converting packed image to planar channels and vice-versa (`nppiCopy_8u_C3P3R`, `nppiCopy_8u_C4P4R`, `nppiCopy_8u_P3C3R`, `nppiCopy_8u_P4C4R`).
  - Converting to grayscale using `nppiColorTwist32f_8u_IP3R` function.

### Miscellaneous

- `make run` will build the project if needed.
- `make USE_NPP=no` will build the project without using CUDA/NPP support. Grayscale conversion will be done on CPU. Not optimized, only for checking purpose.

### Google C++ Style Guide standards...
Code does not meet Google C++ Style Guide standards... I know...
- Lab code from teacher does not meet Google C++ Style Guide standards neither...
- It is not because a big company have its own coding style we should be forced to use it...
- I am not C++ guy... I am old HPC FORTRAN guy... Project as some C++ parts, but it is mostly written in C... Why should I learn and follow a C++ coding style guide?
- Is this C++ coding style courses, or CUDA programming courses?

### Enjoy your review!
