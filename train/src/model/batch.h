//
// Created by adarw on 4/19/26.
//

#pragma once

#include <stdlib.h>
#include <dataset/img_handler/img_handler.h>

typedef struct {
    uint32_t cnt;
    mnist_image_t* images;
    uint8_t* labels;
} batch_t;

uint32_t get_batch_count(mnist_dataset_t* dataset, int batch_size);

batch_t* get_batches(mnist_dataset_t* dataset, int batch_size);

void free_batches(batch_t* batches, uint32_t batch_cnt);
