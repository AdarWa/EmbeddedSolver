//
// Created by adarw on 4/18/26.
//

#pragma once

#include <stdint.h>

#include "tensor/tensor.h"

#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28

typedef struct {
    uint8_t pixels[MNIST_IMAGE_ROWS][MNIST_IMAGE_COLS];
} mnist_raw_image_t;

typedef struct {
    tensor_t* pixels;
} mnist_image_t;

typedef struct {
    uint32_t count;
    mnist_image_t* images;
    uint8_t* labels;
} mnist_dataset_t;

mnist_dataset_t mnist_parse_dataset(const char* image_path, const char* label_path);
void mnist_free_dataset(mnist_dataset_t* dataset);
void mnist_print_image(const mnist_image_t* image);