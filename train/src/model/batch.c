//
// Created by adarw on 4/19/26.
//

#include "batch.h"

#include "logging/logging.h"

static const char* TAG = "model/batch";

uint32_t get_batch_count(mnist_dataset_t* dataset, int batch_size) {
    if (batch_size <= 0) return 0;
    return (dataset->count + batch_size - 1) / batch_size;
}

batch_t* get_batches(mnist_dataset_t* dataset, int batch_size) {
    uint32_t batch_cnt = get_batch_count(dataset, batch_size);
    batch_t* batches = malloc(batch_cnt * sizeof(batch_t));
    if (!batches) {
        LOG_F(TAG, "Couldn't allocate batches array!");
        return NULL;
    }

    for (uint32_t batch = 0; batch < batch_cnt; batch++) {
        batch_t* curr = &batches[batch];

        uint32_t start_idx = batch * batch_size;

        uint32_t current_batch_size = batch_size;
        if (start_idx + batch_size > dataset->count) {
            current_batch_size = dataset->count - start_idx;
        }

        *curr = (batch_t){
            .cnt = current_batch_size,
            .images = malloc(current_batch_size * sizeof(mnist_raw_image_t)),
            .labels = malloc(current_batch_size * sizeof(uint8_t))
        };

        if (!curr->images || !curr->labels) {
            LOG_F(TAG, "Couldn't allocate batch data arrays!");
            return NULL;
        }

        for (uint32_t i = 0; i < current_batch_size; i++) {
            curr->images[i] = dataset->images[start_idx + i];
            curr->labels[i] = dataset->labels[start_idx + i];
        }
    }

    return batches;
}

void free_batches(batch_t* batches, uint32_t batch_cnt) {
    if (batches == NULL) {
        return;
    }

    for (uint32_t i = 0; i < batch_cnt; i++) {
        free(batches[i].images);
        free(batches[i].labels);
    }

    free(batches);
}
