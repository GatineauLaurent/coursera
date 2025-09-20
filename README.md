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

- `Makefile` Top `Makefile` used for compilation, clean-up and run the project. Compilation flags are declared in this file.

### Source code organization

- `src/Makefile` should be called by the top `Makefile`.

- `src/main.cc` Contains the `main` function which is responsible of:
  - Parsing command line arguments.
  - Checking command line arguments correctness and display usage in case of error.
  - Calling `process` function for each input file.

- `src/process.h` Header file containing the prototype of the `process` function.

- `src/process.c` Contains the function `process` function responsible of:
  - Checking input file is a supported image file.
  - Managing host memory for the input and output file.

### Google C++ Style Guide standards...
Code does not meet Google C++ Style Guide standards... I know...
- Lab code from teacher does not meet Google C++ Style Guide standards either...
- It is not because a big company have its own coding style we should be forced to use it...
- I am not C++ guy... I am old HPC FORTRAN guy... Project as some C++ parts, but it is mostly written in C... Why should I learn and follow a C++ coding style guide?
- Is this C++ coding style courses, or CUDA programming courses?

### Enjoy!
