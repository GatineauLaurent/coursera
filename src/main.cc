#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <pthread.h>
#include <unistd.h>

#ifdef _HAVE_NPP
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "process.h"

// Structure to handle threads and exchange data between master process and threads.
typedef struct {
  pthread_t thread;
  int thread_id;
  std::vector<char *> *input_files;
  char *output_dir;
  pthread_mutex_t *mutex_counter;
  int *counter;
} s_thread;

static void *thread_start(void *arg)
{
  s_thread *thread = (s_thread *) arg;
  std::vector<char *> &input_files = *thread->input_files;
  int &counter = *thread->counter; // Counter is shared between threads!

#ifdef _HAVE_NPP
  if (cudaSetDevice(thread->thread_id) != cudaSuccess) {
    std::cout << "Failed to initialize GPU!" << std::endl;
    return NULL;
  }
#endif

  while (1) {
    char *input_file;

    // Get the next file to process
    // Use mutex to access and increment shared counter.
    if (pthread_mutex_lock(thread->mutex_counter)) {
      std::cout << "Failed to lock mutex!" << std::endl;
      return NULL;
    }

    input_file = (counter == input_files.size() ? NULL : input_files[counter++]);
    if (input_file) std::cout << "thread #" << thread->thread_id << " will process file " << input_file << std::endl;

    if (pthread_mutex_unlock(thread->mutex_counter)) {
      std::cout << "Failed to unlock mutex!" << std::endl;
      return NULL;
    }

    if (!input_file) break;
    
    process(input_file, thread->output_dir);
  }

  return NULL;
}

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
  /* Parse arguments, build list of input files, search for output_dir argument... */
  char *output_dir = NULL;
  int input = 0;
  std::vector<char *> input_files;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--help")) usage("", argv[0]);
    if (!strcmp(argv[i], "-h")) usage("", argv[0]);
    if (!strcmp(argv[i], "--output_dir")) {
      if (i + 1 == argc) usage("Missing argument for output_dir option!", argv[0]);
      output_dir = argv[i+1];
      input = 0;
      continue;
    }
    if (!strcmp(argv[i], "--input")) {
      input = 1;
      continue;
    }
    if (input) {
      char *input_file = argv[i];
      if (!std::filesystem::is_regular_file(input_file)) {
        std::cout << "Skip '" << input_file << "' which is not a regular file!" << std::endl;
        continue;
      }
      input_files.push_back(input_file);
    }
  }

  // Check output directory was given and exists
  if (!output_dir) usage("Missing output directory!", argv[0]);
  if (!std::filesystem::is_directory(output_dir)) usage("output directory is not a directory!", argv[0]);

  // Counter and its mutex to process files in multithreaded environment.
  pthread_mutex_t mutex_counter = PTHREAD_MUTEX_INITIALIZER;
  int counter = 0;

#ifdef _HAVE_NPP
  // Use 1 thread per GPU.
  int num_threads = 0;
  cudaGetDeviceCount(&num_threads);
  if (num_threads <= 0) {
    std::cout << "No GPU detected!" << std::endl;
    return 1;
  }
#else
  // Use 1 thread per CPU core.
  int num_threads = (int) sysconf(_SC_NPROCESSORS_ONLN);
#endif

  s_thread *threads = (s_thread *) malloc(sizeof(s_thread) * num_threads);
  if (!threads) {
    std::cout << "Failed to allocate threads!" << std::endl;
    return 1;
  }

  // Start threads.
  for (int i = 0; i < num_threads; i++) {
    threads[i].thread_id = i;
    threads[i].input_files = &input_files;
    threads[i].mutex_counter = &mutex_counter;
    threads[i].counter = &counter;
    threads[i].output_dir = output_dir;
    if (pthread_create(&threads[i].thread, NULL, thread_start, &threads[i])) {
      std::cout << "Failed to start threads #" << i << "!" << std::endl;
    }
  }

  // Wait for all threads.
  for (int i = 0; i < num_threads; i++) {
    if (pthread_join(threads[i].thread, NULL)) {
      std::cout << "Failed to join threads #" << i << "!" << std::endl;
    }
  }

  free(threads);

  return 0;
}
