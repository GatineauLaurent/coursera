#include <iostream>
#include <filesystem>
#include <cstring>
#include <FreeImage.h>

#include "process.h"

void process(char *input_file, char *output_dir)
{
  std::cout << "Process file \"" << input_file << "\"..." << std::endl;

  // Check if file format is supported by FreeImage, and if file can be read.
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(input_file, 0);
  if (fif == FIF_UNKNOWN) {
    std::cout << "  File format not supported by FreeImage!" << std::endl;
    return;
  }

  // Load input file into host memory.
  FIBITMAP *input_bitmap = FreeImage_Load(fif, input_file, 0);
  if (!input_bitmap) {
    std::cout << "  Failed to load file using FreeImage!" << std::endl;
    return;
  }

  // Get input image information.
  FREE_IMAGE_TYPE input_fit = FreeImage_GetImageType(input_bitmap);
  unsigned int input_bits_per_pixel = FreeImage_GetBPP(input_bitmap);
  unsigned int input_width = FreeImage_GetWidth(input_bitmap); // In pixels
  unsigned int input_height = FreeImage_GetHeight(input_bitmap); // In pixels
  unsigned int input_stride = FreeImage_GetPitch(input_bitmap); // In bytes (can include padding)

  // Allocate output image. 
  FIBITMAP *output_bitmap = FreeImage_Allocate(input_width, input_height, input_bits_per_pixel);
  if (!output_bitmap) {
    std::cout << "  Failed to allocate output image in host memory!" << std::endl;
    FreeImage_Unload(input_bitmap);
    return;
  }

  // Convert to grayscale...

  // Save output image, using same format as input file, but converted to grayscale
  std::filesystem::path output_file = output_dir;
  output_file /= std::filesystem::path(input_file).filename();
  if (!FreeImage_Save(fif, output_bitmap, output_file.string().c_str(), 0)) {
    std::cout << "  Failed to save output file " << output_file << "!" << std::endl;
  } else {
    std::cout << "  Successfull converted to grayscale and saved into " << output_file << "!" << std::endl;
  }
  
  // Free input and output image from host memory.
  FreeImage_Unload(input_bitmap);
  FreeImage_Unload(output_bitmap);
}
