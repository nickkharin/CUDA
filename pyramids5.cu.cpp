#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Объявления функций CUDA

// Ядро CUDA для гауссова размытия
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int size = 1; // Размер окна фильтра (здесь простое усреднение)

    for (int dy = -size; dy <= size; dy++) {
        for (int dx = -size; dx <= size; dx++) {
            int ix = min(max(x + dx, 0), width - 1);
            int iy = min(max(y + dy, 0), height - 1);

            sum += input[iy * width + ix];
        }
    }

    output[y * width + x] = static_cast<unsigned char>(sum / ((2 * size + 1) * (2 * size + 1)));
}

__global__ void combinedBlurDownsampleKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= inputWidth || y >= inputHeight) return;

    float sum = 0.0f;
    int size = 1; // Размер окна фильтра

    for (int dy = -size; dy <= size; dy++) {
        for (int dx = -size; dx <= size; dx++) {
            int ix = min(max(x + dx, 0), inputWidth - 1);
            int iy = min(max(y + dy, 0), inputHeight - 1);

            sum += input[iy * inputWidth + ix];
        }
    }
    sum /= ((2 * size + 1) * (2 * size + 1));

    // Уменьшаем размер изображения, выбирая каждый второй пиксель
    if (x % 2 == 0 && y % 2 == 0) {
        output[(y / 2) * outputWidth + (x / 2)] = static_cast<unsigned char>(sum);
    }
}



// Ядро CUDA для увеличения размера изображения
__global__ void upsampleImageKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= inputWidth * 2 || y >= inputHeight * 2) return;

    output[y * (inputWidth * 2) + x] = input[(y / 2) * inputWidth + (x / 2)];
}


// Ядро CUDA для вычитания двух изображений
__global__ void subtractImagesKernel(unsigned char* img1, unsigned char* img2, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;
    int diff = static_cast<int>(img1[index]) - static_cast<int>(img2[index]);
    output[index] = static_cast<unsigned char>(max(0, min(255, diff)));
}


#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

void gaussianBlur(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char *dev_input, *dev_output;

    cudaMalloc(&dev_input, width * height * sizeof(unsigned char));
    cudaCheckError();

    cudaMemcpy(dev_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaMalloc(&dev_output, width * height * sizeof(unsigned char));
    cudaCheckError();


    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlurKernel<<<gridSize, blockSize>>>(dev_input, dev_output, width, height);

    cudaMemcpy(output, dev_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(dev_input);
    cudaCheckError();

    cudaFree(dev_output);
    cudaCheckError();

}

// Оболочки функций для вызова ядер CUDA
void gaussianBlurDownsample(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    unsigned char *dev_input, *dev_output;

    cudaMalloc(&dev_input, inputWidth * inputHeight * sizeof(unsigned char));
    cudaMemcpy(dev_input, input, inputWidth * inputHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_output, outputWidth * outputHeight * sizeof(unsigned char));

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

    combinedBlurDownsampleKernel<<<gridSize, blockSize>>>(dev_input, dev_output, inputWidth, inputHeight, outputWidth, outputHeight);

    cudaMemcpy(output, dev_output, outputWidth * outputHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
}

// Ядро CUDA для уменьшения размера изображения
__global__ void downsampleImageKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= inputWidth / 2 || y >= inputHeight / 2) return;

    output[y * (inputWidth / 2) + x] = input[(2 * y) * inputWidth + (2 * x)];
}

void upsampleImage(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight) {
    unsigned char *dev_input, *dev_output;

    cudaMalloc(&dev_input, inputWidth * inputHeight * sizeof(unsigned char));
    cudaMemcpy(dev_input, input, inputWidth * inputHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_output, inputWidth * 2 * inputHeight * 2 * sizeof(unsigned char));

    dim3 blockSize(16, 16);
    dim3 gridSize((inputWidth * 2 + blockSize.x - 1) / blockSize.x, (inputHeight * 2 + blockSize.y - 1) / blockSize.y);

    upsampleImageKernel<<<gridSize, blockSize>>>(dev_input, dev_output, inputWidth, inputHeight);

    cudaMemcpy(output, dev_output, inputWidth * 2 * inputHeight * 2 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
}

void subtractImages(unsigned char* img1, unsigned char* img2, unsigned char* output, int width, int height) {
    unsigned char *dev_img1, *dev_img2, *dev_output;

    cudaMalloc(&dev_img1, width * height * sizeof(unsigned char));
    cudaMemcpy(dev_img1, img1, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_img2, width * height * sizeof(unsigned char));
    cudaMemcpy(dev_img2, img2, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_output, width * height * sizeof(unsigned char));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    subtractImagesKernel<<<gridSize, blockSize>>>(dev_img1, dev_img2, dev_output, width, height);

    cudaMemcpy(output, dev_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(dev_img1);
    cudaFree(dev_img2);
    cudaFree(dev_output);
}

// Функция для сложения двух изображений
void addImages(unsigned char* img1, unsigned char* img2, unsigned char* output, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        int sum = static_cast<int>(img1[i]) + static_cast<int>(img2[i]);
        output[i] = static_cast<unsigned char>(std::min(255, sum));
    }
}

// Дополнительные функции
void createGaussianPyramid(unsigned char* input, std::vector<unsigned char*>& pyramid, int levels, int width, int height) {
    for (int i = 0; i < levels; i++) {
        int outputWidth = width >> i;
        int outputHeight = height >> i;
        pyramid[i] = new unsigned char[outputWidth * outputHeight];
        if (i == 0) {
            // Первый уровень - просто применяем гауссово размытие
            gaussianBlur(input, pyramid[i], width, height); 
        } else {
            // На последующих уровнях - уменьшаем размер
            gaussianBlurDownsample(pyramid[i - 1], pyramid[i], width >> (i - 1), height >> (i - 1), outputWidth, outputHeight);
        }
    }
}

void createLaplacianPyramid(std::vector<unsigned char*>& gaussianPyramid, std::vector<unsigned char*>& laplacianPyramid, int levels, int width, int height) {
    for (int i = 0; i < levels - 1; i++) {
        laplacianPyramid[i] = new unsigned char[(width >> i) * (height >> i)];
        unsigned char* upsampled = new unsigned char[(width >> i) * (height >> i)];
        upsampleImage(gaussianPyramid[i + 1], upsampled, width >> (i + 1), height >> (i + 1));
        subtractImages(gaussianPyramid[i], upsampled, laplacianPyramid[i], width >> i, height >> i);
        delete[] upsampled;
    }
    laplacianPyramid[levels - 1] = gaussianPyramid[levels - 1];
}

void reconstructFromLaplacianPyramid(std::vector<unsigned char*>& laplacianPyramid, unsigned char* output, int levels, int width, int height) {
    for (int i = levels - 1; i > 0; --i) {
        unsigned char* upsampled = new unsigned char[(width >> (i - 1)) * (height >> (i - 1))];
        upsampleImage(laplacianPyramid[i], upsampled, width >> i, height >> i);
        addImages(upsampled, laplacianPyramid[i - 1], laplacianPyramid[i - 1], width >> (i - 1), height >> (i - 1));
        delete[] upsampled;
    }
    memcpy(output, laplacianPyramid[0], width * height * sizeof(unsigned char));
}


int main() {
    std::cout << "Program started" << std::endl;
    int width, height, channels;
    unsigned char* img = stbi_load("grayscale.png", &width, &height, &channels, 0);
    if (img == nullptr) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    std::cout << "Loaded image with dimensions: " << width << "x" << height << ", Channels: " << channels << std::endl;

    // Проверяем, что изображение одноканальное (градации серого)
    if (channels != 1) {
        std::cerr << "The image must be a grayscale image" << std::endl;
        stbi_image_free(img);
        return -1;
    }

    const int levels = 3; // Количество уровней в пирамиде

    std::vector<unsigned char*> gaussianPyramid(levels);
    std::vector<unsigned char*> laplacianPyramid(levels);

    createGaussianPyramid(img, gaussianPyramid, levels, width, height);
    stbi_write_png("gaussian_level0.png", width, height, 1, gaussianPyramid[0], width);

    for (int i = 1; i < levels; ++i) {
        char filename[64];
        sprintf(filename, "gaussian_level%d.png", i);
        int levelWidth = width >> i;
        int levelHeight = height >> i;
        stbi_write_png(filename, levelWidth, levelHeight, 1, gaussianPyramid[i], levelWidth);
    }


    createLaplacianPyramid(gaussianPyramid, laplacianPyramid, levels, width, height);

    for (int i = 0; i < levels; ++i) {
        char filename[64];
        sprintf(filename, "laplacian_level%d.png", i);
        int levelWidth = width >> i;
        int levelHeight = height >> i;
        stbi_write_png(filename, levelWidth, levelHeight, 1, laplacianPyramid[i], levelWidth);
    }

    // Реконструкция изображения из Лапласианской пирамиды
    unsigned char* reconstructed = new unsigned char[width * height];
    reconstructFromLaplacianPyramid(laplacianPyramid, reconstructed, levels, width, height);

    // Сохранение изображения на диск
    if (stbi_write_png("reconstructed_image.png", width, height, 1, reconstructed, width)) {
        std::cout << "Reconstructed image saved as reconstructed_image.png" << std::endl;
    } else {
        std::cerr << "Failed to save the reconstructed image." << std::endl;
    }

    // Освобождение памяти
    stbi_image_free(img);
    for (auto& ptr : gaussianPyramid) delete[] ptr;
    for (auto& ptr : laplacianPyramid) {
        if (&ptr != &gaussianPyramid.back()) {
            delete[] ptr;
        }
    }
    delete[] reconstructed;
}
