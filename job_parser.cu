#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <png.h>

#define BLOCK_SIZE 256

// Timer utility
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// PNG Loader
unsigned char* load_png(const char* filename, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Error: Could not open %s\n", filename); return NULL; }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return NULL; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return NULL; }

    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return NULL; }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 255, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    int rowbytes = png_get_rowbytes(png, info);
    unsigned char* image_data = (unsigned char*)malloc(rowbytes * *height);
    if (!image_data) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return NULL; }

    png_bytep row_pointers[*height];
    for (int i = 0; i < *height; i++) row_pointers[i] = image_data + i * rowbytes;

    png_read_image(png, row_pointers);

    png_read_end(png, info);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return image_data;
}

// PNG Saver
void save_png(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "Error: Could not save %s\n", filename); return; }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, NULL); fclose(fp); return; }

    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return; }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep row_pointers[height];
    for (int i = 0; i < height; i++) row_pointers[i] = image + i * width * 3;

    png_write_image(png, row_pointers);
    png_write_end(png, info);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

// Serial Functions
void brightness_serial(unsigned char* image, int width, int height, int brightness) {
    for (int i = 0; i < width * height * 3; i++) {
        int temp = image[i] + brightness;
        image[i] = (temp < 0) ? 0 : (temp > 255 ? 255 : temp);
    }
}

void grayscale_serial(unsigned char* image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int r = image[i * 3], g = image[i * 3 + 1], b = image[i * 3 + 2];
        int gray = 0.299 * r + 0.587 * g + 0.114 * b;
        image[i * 3] = image[i * 3 + 1] = image[i * 3 + 2] = gray;
    }
}

void blur_serial(unsigned char* image, int width, int height) {
    unsigned char* temp = (unsigned char*)malloc(width * height * 3);
    memcpy(temp, image, width * height * 3);
    for (int i = 1; i < width * height * 3 - 1; i++) {
        image[i] = (temp[i - 1] + temp[i] + temp[i + 1]) / 3;
    }
    free(temp);
}

void sharpen_serial(unsigned char* image, int width, int height) {
    unsigned char* temp = (unsigned char*)malloc(width * height * 3);
    memcpy(temp, image, width * height * 3);
    for (int i = 1; i < width * height * 3 - 1; i++) {
        int sharp = (5 * temp[i]) - (temp[i - 1] + temp[i + 1]);
        image[i] = (sharp < 0) ? 0 : (sharp > 255 ? 255 : sharp);
    }
    free(temp);
}


// CUDA Kernels
__global__ void brightness_kernel(unsigned char* image, int width, int height, int brightness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * 3) {
        int temp = image[idx] + brightness;
        image[idx] = min(max(temp, 0), 255);
    }
}

__global__ void grayscale_kernel(unsigned char* image, int width, int height, unsigned char* temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int r = image[idx * 3], g = image[idx * 3 + 1], b = image[idx * 3 + 2];
        int gray = 0.299 * r + 0.587 * g + 0.114 * b;
        image[idx * 3] = image[idx * 3 + 1] = image[idx * 3 + 2] = gray;
    }
}

__global__ void blur_kernel(unsigned char* image, int width, int height, unsigned char* temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < width * height * 3 - 1) {
        image[idx] = (temp[idx - 1] + temp[idx] + temp[idx + 1]) / 3;
    }
}

__global__ void sharpen_kernel(unsigned char* image, int width, int height, unsigned char* temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < width * height * 3 - 1) {
        int sharp = (5 * temp[idx]) - (temp[idx - 1] + temp[idx + 1]);
        image[idx] = min(max(sharp, 0), 255);
    }
}


// Main program
int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image.png>\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    int width, height;
    unsigned char* image = load_png(input_file, &width, &height);
    if (!image) {
        fprintf(stderr, "Error loading image: %s\n", input_file);
        return 1;
    }

    int imageSize = width * height * 3;
    unsigned char* d_image, *d_temp;
    cudaError_t err = cudaMalloc(&d_image, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc Error (d_image): %s\n", cudaGetErrorString(err));
        free(image);
        return 1;
    }
    err = cudaMalloc(&d_temp, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc Error (d_temp): %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(image);
        return 1;
    }

    FILE* outputFile = fopen("execution_times.txt", "w");
    if (!outputFile) {
        fprintf(stderr, "Could not open execution_times.txt\n");
        cudaFree(d_image);
        cudaFree(d_temp);
        free(image);
        return 1;
    }

    struct Operation {
        const char* name;
        void (*serial_func3)(unsigned char*, int, int);
        void (*serial_func4)(unsigned char*, int, int, int);
        void (*cuda_kernel)(unsigned char*, int, int, unsigned char*);
        const char* output_name;
        bool per_pixel;
    };

    struct Operation operations[] = {
        {"Brightness", NULL, brightness_serial, (void (*)(unsigned char*, int, int, unsigned char*))brightness_kernel, "brightness_out.png", false},
        {"Grayscale", grayscale_serial, NULL, (void (*)(unsigned char*, int, int, unsigned char*))grayscale_kernel, "grayscale_out.png", true},
        {"Blurring", blur_serial, NULL, blur_kernel, "blur_out.png", false},
        {"Sharpening", sharpen_serial, NULL, sharpen_kernel, "sharpen_out.png", false},
    };

    int brightness = 50;

    // Print CUDA device info for debugging
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Running on GPU: %s (Compute Capability: %d.%d)\n", prop.name, prop.major, prop.minor);

    for (int i = 0; i < 5; i++) {
        // Serial execution
        unsigned char* serial_image = (unsigned char*)malloc(imageSize);
        if (!serial_image) {
            fprintf(stderr, "Error: Failed to allocate serial_image\n");
            fclose(outputFile);
            cudaFree(d_image);
            cudaFree(d_temp);
            free(image);
            return 1;
        }
        memcpy(serial_image, image, imageSize);

        double start_serial = get_time();
        if (operations[i].serial_func4)
            operations[i].serial_func4(serial_image, width, height, brightness);
        else
            operations[i].serial_func3(serial_image, width, height);
        double end_serial = get_time();

        save_png(operations[i].output_name, serial_image, width, height);
        fprintf(outputFile, "%s Serial: %f ms\n", operations[i].name, end_serial - start_serial);
        free(serial_image);

        // CUDA execution
        err = cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Memcpy Error (host to device, %s): %s\n", operations[i].name, cudaGetErrorString(err));
            continue;
        }
        err = cudaMemcpy(d_temp, image, imageSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Memcpy Error (temp, %s): %s\n", operations[i].name, cudaGetErrorString(err));
            continue;
        }

        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize = operations[i].per_pixel ?
            dim3((width * height + blockSize.x - 1) / blockSize.x) :
            dim3((imageSize + blockSize.x - 1) / blockSize.x);

        double start_parallel = get_time();
        operations[i].cuda_kernel<<<gridSize, blockSize>>>(d_image, width, height, operations[i].per_pixel ? NULL : d_temp);
        err = cudaDeviceSynchronize();
        double end_parallel = get_time();

        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in %s: %s\n", operations[i].name, cudaGetErrorString(err));
            continue;
        }

        // Save CUDA output
        unsigned char* cuda_image = (unsigned char*)malloc(imageSize);
        if (!cuda_image) {
            fprintf(stderr, "Error: Failed to allocate cuda_image\n");
            continue;
        }
        err = cudaMemcpy(cuda_image, d_image, imageSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Memcpy Error (device to host, %s): %s\n", operations[i].name, cudaGetErrorString(err));
            free(cuda_image);
            continue;
        }
        char cuda_output_name[256];
        snprintf(cuda_output_name, sizeof(cuda_output_name), "cuda_%s", operations[i].output_name);
        save_png(cuda_output_name, cuda_image, width, height);
        free(cuda_image);

        fprintf(outputFile, "%s Parallel: %f ms\n", operations[i].name, end_parallel - start_parallel);
        fflush(outputFile);
    }

    printf("Done. Execution times saved to execution_times.txt\n");

    fclose(outputFile);
    cudaFree(d_image);
    cudaFree(d_temp);
    free(image);

    return 0;
}
