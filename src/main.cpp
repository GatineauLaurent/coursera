#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <cstring>

#include "process.h"

static void usage(char const *msg, char const *prg)
{
  std::cout << msg << std::endl;
  std::cout << std::endl;
  std::cout << prg << " [--help|-h] --input [file1|file2|...] --output_dir output" << std::endl;
  std::cout << "   --input: list of input files" << std::endl;
  std::cout << "   --output_dir: output directory (it must exist!)" << std::endl;
  std::cout << std::endl;
  std::cout << "Convert all input files to grayscale and put results in output directory." << std::endl;
  std::cout << "!!!Files in output directory will be overwritten!!!" << std::endl;
  exit(1);
}

int main(int argc, char *argv[])
{
  char *output_dir = NULL;

  /* Search for output_dir argument, and check if directory exists */
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--help")) usage("", argv[0]);
    if (!strcmp(argv[i], "-h")) usage("", argv[0]);
    if (!strcmp(argv[i], "--output_dir")) {
      if (i + 1 == argc) usage("Missing argument for output_dir option!", argv[0]);
      output_dir = argv[i+1];
      continue;
    }
  }

  if (!output_dir) usage("Missing output directory!", argv[0]);
  if (!std::filesystem::is_directory(output_dir)) usage("output directory is not a directory!", argv[0]);

  /* Process all input files */
  int input = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--input")) {
      input = 1;
      continue;
    }
    if (!strcmp(argv[i], "--output_dir")) {
      input = 0;
      continue;
    }
    if (input) {
      char *input_file = argv[i];
      if (!std::filesystem::is_regular_file(input_file)) {
        std::cout << "Skip '" << input_file << "' which is not a regular file!" << std::endl;
        continue;
      }
      process(input_file);
    }
  }

  return 0;
}
