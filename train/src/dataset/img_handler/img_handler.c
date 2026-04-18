//
// Created by adarw on 4/18/26.
//

#include "img_handler.h"

#include <stdio.h>
#include <stdlib.h>

#include "logging/logging.h"

static const char* TAG = "mnist_parser";

// MNIST magic numbers
#define LABEL_MAGIC 2049
#define IMAGE_MAGIC 2051

static uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) | ((uint32_t)c2 << 16) | ((uint32_t)c3 << 8) | c4;
}

static uint8_t* parse_labels(const char* path, uint32_t* out_count) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        LOG_E(TAG, "Failed to open label file: %s", path);
        return NULL;
    }

    uint32_t magic = 0;
    uint32_t count = 0;

    fread(&magic, sizeof(uint32_t), 1, file);
    magic = reverse_int(magic);
    if (magic != LABEL_MAGIC) {
        LOG_E(TAG, "Invalid magic number in label file %s: Expected %d, got %d", path, LABEL_MAGIC, magic);
        fclose(file);
        return NULL;
    }

    fread(&count, sizeof(uint32_t), 1, file);
    count = reverse_int(count);
    *out_count = count;

    LOG_D(TAG, "Parsing %u labels from %s", count, path);

    uint8_t* labels = (uint8_t*)malloc(count * sizeof(uint8_t));
    if (!labels) {
        LOG_E(TAG, "Failed to allocate memory for labels");
        fclose(file);
        return NULL;
    }

    fread(labels, sizeof(uint8_t), count, file);
    fclose(file);
    return labels;
}

static mnist_image_t* parse_images(const char* path, uint32_t* out_count) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        LOG_E(TAG, "Failed to open image file: %s", path);
        return NULL;
    }

    uint32_t magic = 0;
    uint32_t count = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    fread(&magic, sizeof(uint32_t), 1, file);
    magic = reverse_int(magic);
    if (magic != IMAGE_MAGIC) {
        LOG_E(TAG, "Invalid magic number in image file %s: Expected %d, got %d", path, IMAGE_MAGIC, magic);
        fclose(file);
        return NULL;
    }

    fread(&count, sizeof(uint32_t), 1, file);
    count = reverse_int(count);
    *out_count = count;

    fread(&rows, sizeof(uint32_t), 1, file);
    rows = reverse_int(rows);

    fread(&cols, sizeof(uint32_t), 1, file);
    cols = reverse_int(cols);

    if (rows != MNIST_IMAGE_ROWS || cols != MNIST_IMAGE_COLS) {
        LOG_E(TAG, "Unexpected image dimensions: %dx%d. Expected %dx%d", rows, cols, MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS);
        fclose(file);
        return NULL;
    }

    LOG_D(TAG, "Parsing %u images (%dx%d) from %s", count, rows, cols, path);

    mnist_image_t* images = (mnist_image_t*)malloc(count * sizeof(mnist_image_t));
    if (!images) {
        LOG_E(TAG, "Failed to allocate memory for images");
        fclose(file);
        return NULL;
    }

    // Read all pixel data directly into the structs
    fread(images, sizeof(mnist_image_t), count, file);
    fclose(file);
    return images;
}

mnist_dataset_t mnist_parse_dataset(const char* image_path, const char* label_path) {
    mnist_dataset_t dataset = {0, NULL, NULL};

    uint32_t label_count = 0;
    uint32_t image_count = 0;

    dataset.labels = parse_labels(label_path, &label_count);
    if (!dataset.labels) {
        return dataset; // Return empty/failed dataset
    }

    dataset.images = parse_images(image_path, &image_count);
    if (!dataset.images) {
        free(dataset.labels);
        dataset.labels = NULL;
        return dataset;
    }

    // Sanity check to ensure images and labels align
    if (image_count != label_count) {
        LOG_E(TAG, "Mismatch: %u images but %u labels", image_count, label_count);
        free(dataset.labels);
        free(dataset.images);
        dataset.labels = NULL;
        dataset.images = NULL;
        return dataset;
    }

    dataset.count = image_count;
    LOG_D(TAG, "Successfully parsed dataset. Total items: %u", dataset.count);
    return dataset;
}

void mnist_free_dataset(mnist_dataset_t* dataset) {
    if (dataset) {
        if (dataset->images) {
            free(dataset->images);
            dataset->images = NULL;
        }
        if (dataset->labels) {
            free(dataset->labels);
            dataset->labels = NULL;
        }
        dataset->count = 0;
    }
}

void mnist_print_image(const mnist_image_t* image) {
    if (!image) {
        LOG_E(TAG, "Cannot print null image.");
        return;
    }

    // Top border
    printf("========================================================\n");

    for (int r = 0; r < MNIST_IMAGE_ROWS; r++) {
        for (int c = 0; c < MNIST_IMAGE_COLS; c++) {
            uint8_t pixel = image->pixels[r][c];
            char ch = ' ';

            // Map pixel intensity (0-255) to ASCII characters
            if (pixel > 200) {
                ch = '#';
            } else if (pixel > 128) {
                ch = '*';
            } else if (pixel > 64) {
                ch = '.';
            }

            // Print character twice to correct terminal aspect ratio
            putchar(ch);
            putchar(ch);
        }
        putchar('\n');
    }

    // Bottom border
    printf("========================================================\n");
}