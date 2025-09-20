#include <iostream>
#include <filesystem>
#include <cstring>
#include <FreeImage.h>

#ifdef _HAVE_NPP
#include <npp.h>
#endif

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
  if (input_fit != FIT_BITMAP) {
    std::cout << "  Image type not supported (" << input_fit << "!= FIT_BITMAP (" << FIT_BITMAP << "))!" << std::endl;
    FreeImage_Unload(input_bitmap);
    return;
  }
  unsigned int input_bits_per_pixel = FreeImage_GetBPP(input_bitmap);
  if (input_bits_per_pixel != 24 && input_bits_per_pixel != 32) {
    std::cout << "  Bits per pixel not supported (" << input_bits_per_pixel << " != 24 or 32)" << std::endl;
    FreeImage_Unload(input_bitmap);
    return;
  }
  size_t input_width = FreeImage_GetWidth(input_bitmap); // In pixels
  size_t input_height = FreeImage_GetHeight(input_bitmap); // In pixels
  size_t input_stride = FreeImage_GetPitch(input_bitmap); // In bytes (can include padding)

  // Allocate output image on host memory. 
  FIBITMAP *output_bitmap = FreeImage_Allocate(input_width, input_height, input_bits_per_pixel);
  if (!output_bitmap) {
    std::cout << "  Failed to allocate output image in host memory!" << std::endl;
    FreeImage_Unload(input_bitmap);
    return;
  }

#ifdef _HAVE_NPP
  // Convert freeimage RGB/RGBA packed format to 3/4 channels NPP format.
  // Convert to grayscale using color twist NPP functions.
  // Convert back to freeimage RGB/RGBA packed format.

  cudaError_t err;
  void *dev_fi;
  size_t size = input_stride * input_height;

  // Allocate memory on device for the input image.
  err = cudaMalloc(&dev_fi, size);
  if (err != cudaSuccess) {
    std::cout << "  Failed to allocate device memory!" << std::endl;
    FreeImage_Unload(input_bitmap);
    FreeImage_Unload(output_bitmap);
    return;
  }

  // Copy input image to the device.
  err = cudaMemcpy(dev_fi, FreeImage_GetBits(input_bitmap), size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cout << "  Failed to copy image to the device!" << std::endl;
    FreeImage_Unload(input_bitmap);
    FreeImage_Unload(output_bitmap);
    cudaFree(dev_fi);
    return;
  }

  // Allocate NPP plans on the device
  Npp8u *dev_plans[4];
  int plan_step;

  for (int i = 0; i < (input_bits_per_pixel == 24 ? 3 : 4); i++) {
    dev_plans[i] = nppiMalloc_8u_C1(input_width, input_height, &plan_step);
    if (dev_plans[i] == NULL) {
      std::cout << "  Failed to allocate device memory!" << std::endl;
      FreeImage_Unload(input_bitmap);
      FreeImage_Unload(output_bitmap);
      cudaFree(dev_fi);
      for (int j = 0; j < i - 1; j++) nppiFree(dev_plans[j]);
      return;
    }
  }

  // Convert FreeImage to NPP plans (packed to planar channel)
  NppStatus rc;
  NppiSize roi = {(int) input_width, (int) input_height};
  if (input_bits_per_pixel == 24) {
    rc = nppiCopy_8u_C3P3R((Npp8u *) dev_fi, input_stride, dev_plans, plan_step, roi);
  } else {
    rc = nppiCopy_8u_C4P4R((Npp8u *) dev_fi, input_stride, dev_plans, plan_step, roi);
  }
  if (rc != NPP_SUCCESS) {
    std::cout << "  Failed to convert packed image to planar channels!" << std::endl;
    FreeImage_Unload(input_bitmap);
    FreeImage_Unload(output_bitmap);
    cudaFree(dev_fi);
    for(int i = 0; i < (input_bits_per_pixel == 24 ? 3 : 4); i++) nppiFree(dev_plans[i]);
    return;
  }

  // NppStatus 	nppiColorTwist32f_8u_C4IR (Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])

  // Convert NPP plans to FreeImage (planar channel to packed)
  if (input_bits_per_pixel == 24) {
    rc = nppiCopy_8u_P3C3R(dev_plans, plan_step, (Npp8u *)dev_fi, input_stride, roi);
  } else {
    rc = nppiCopy_8u_P4C4R(dev_plans, plan_step, (Npp8u *)dev_fi, input_stride, roi);
  }
  if (rc != NPP_SUCCESS) {
    std::cout << "  Failed to convert planar channels to packed image!" << std::endl;
    FreeImage_Unload(input_bitmap);
    FreeImage_Unload(output_bitmap);
    cudaFree(dev_fi);
    for (int i = 0; i < (input_bits_per_pixel == 24 ? 3 : 4); i++) nppiFree(dev_plans[i]);
    return;
  }

  // Copy device image to output on host.
  err = cudaMemcpy(FreeImage_GetBits(output_bitmap), dev_fi, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    std::cout << "  Failed to copy device image to the host!" << std::endl;
    FreeImage_Unload(input_bitmap);
    FreeImage_Unload(output_bitmap);
    cudaFree(dev_fi);
    for (int i = 0; i < (input_bits_per_pixel == 24 ? 3 : 4); i++) nppiFree(dev_plans[i]);
    return;
  }

  cudaFree(dev_fi);
  for (int i = 0; i < (input_bits_per_pixel == 24 ? 3 : 4); i++) nppiFree(dev_plans[i]);
#else
  // CPU non optimized conversion to grayscale.
  // RGB to grayscale conversion NTSC formula: 0.299 * Red + 0.587 * Green + 0.114 * Blue
  for (int y = 0; y < input_height; y++) {
    unsigned char *input_bits = FreeImage_GetScanLine(input_bitmap, y);
    unsigned char *output_bits = FreeImage_GetScanLine(output_bitmap, y);
    for (int x = 0; x < input_width; x++) {
      float r, g, b, gray;
      r = input_bits[FI_RGBA_RED];
      g = input_bits[FI_RGBA_GREEN];
      b = input_bits[FI_RGBA_BLUE];
      gray = 0.229f * r + 0.587f * g + 0.114f * b;
      if (gray > 255.0f) gray = 255.0f;
      output_bits[FI_RGBA_RED] = gray;
      output_bits[FI_RGBA_GREEN] = gray;
      output_bits[FI_RGBA_BLUE] = gray;
      if (input_bits_per_pixel == 32) {
        output_bits[FI_RGBA_ALPHA] = input_bits[FI_RGBA_ALPHA];
        input_bits += 4;
        output_bits += 4;
      } else {
        input_bits += 3;
        output_bits += 3;
      }
    }
  }
#endif

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
