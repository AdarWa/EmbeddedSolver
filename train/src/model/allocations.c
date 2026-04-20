//
// Created by adarw on 4/19/26.
//

// WTF is this shit
// Fucking hate handling the heap

#include "allocations.h"
#include "logging/logging.h"
#include <stdlib.h>

static const char* TAG = "model/allocations";

weights_t allocate_weights(model_train_config_t* config) {
    weights_t w = {0};

    w.conv_parameters_1.count = config->filter_cnt_1;
    w.conv_parameters_2.count = config->filter_cnt_2;
    w.dense_parameters_1.count = config->dense_cnt_1;
    w.dense_parameters_2.count = config->dense_cnt_2;
    w.dense_parameters_3.count = config->dense_cnt_3;

    w.conv_parameters_1.weights = calloc(w.conv_parameters_1.count, sizeof(tensor_t*));
    w.conv_parameters_1.biases = calloc(w.conv_parameters_1.count, sizeof(double));

    w.conv_parameters_2.weights = calloc(w.conv_parameters_2.count, sizeof(tensor_t*));
    w.conv_parameters_2.biases = calloc(w.conv_parameters_2.count, sizeof(double));

    w.dense_parameters_1.weights = calloc(w.dense_parameters_1.count, sizeof(tensor_t*));
    w.dense_parameters_1.biases = calloc(w.dense_parameters_1.count, sizeof(double));

    w.dense_parameters_2.weights = calloc(w.dense_parameters_2.count, sizeof(tensor_t*));
    w.dense_parameters_2.biases = calloc(w.dense_parameters_2.count, sizeof(double));

    w.dense_parameters_3.weights = calloc(w.dense_parameters_3.count, sizeof(tensor_t*));
    w.dense_parameters_3.biases = calloc(w.dense_parameters_3.count, sizeof(double));

    if (!w.conv_parameters_1.weights || !w.conv_parameters_1.biases ||
        !w.conv_parameters_2.weights || !w.conv_parameters_2.biases ||
        !w.dense_parameters_1.weights || !w.dense_parameters_1.biases ||
        !w.dense_parameters_2.weights || !w.dense_parameters_2.biases ||
        !w.dense_parameters_3.weights || !w.dense_parameters_3.biases) {

        LOG_F(TAG, "Couldn't allocate arrays for weights or biases!");
        free_weights(&w);
        return (weights_t){0};
    }

    // Conv 1
    for (int i = 0; i < w.conv_parameters_1.count; i++) {
        w.conv_parameters_1.weights[i] = allocate_tensor((int[]){config->kernel_size, config->kernel_size, config->channel_cnt}, 3);
        if (!w.conv_parameters_1.weights[i]) goto tensor_alloc_err;
    }

    // Conv 2
    for (int i = 0; i < w.conv_parameters_2.count; i++) {
        w.conv_parameters_2.weights[i] = allocate_tensor((int[]){config->kernel_size, config->kernel_size, config->channel_cnt}, 3);
        if (!w.conv_parameters_2.weights[i]) goto tensor_alloc_err;
    }

    // Dense 1
    for (int i = 0; i < w.dense_parameters_1.count; i++) {
        w.dense_parameters_1.weights[i] = allocate_tensor((int[]){config->dense_cnt_1, config->dense_cnt_1, config->channel_cnt}, 3);
        if (!w.dense_parameters_1.weights[i]) goto tensor_alloc_err;
    }

    // Dense 2
    for (int i = 0; i < w.dense_parameters_2.count; i++) {
        w.dense_parameters_2.weights[i] = allocate_tensor((int[]){config->dense_cnt_2, config->dense_cnt_2, config->channel_cnt}, 3);
        if (!w.dense_parameters_2.weights[i]) goto tensor_alloc_err;
    }

    // Dense 3
    for (int i = 0; i < w.dense_parameters_3.count; i++) {
        w.dense_parameters_3.weights[i] = allocate_tensor((int[]){config->dense_cnt_3, config->dense_cnt_3, config->channel_cnt}, 3);
        if (!w.dense_parameters_3.weights[i]) goto tensor_alloc_err;
    }

    return w;

tensor_alloc_err:
    // catches failures without leaking memory
    LOG_F(TAG, "Couldn't allocate tensor data for a layer!");
    free_weights(&w);
    return (weights_t){0};
}

void free_weights(weights_t* weights) {
    if (!weights) return;

    if (weights->conv_parameters_1.weights) {
        for (int i = 0; i < weights->conv_parameters_1.count; i++) {
            if (weights->conv_parameters_1.weights[i]) free_tensor(weights->conv_parameters_1.weights[i]);
        }
        free(weights->conv_parameters_1.weights);
    }
    if (weights->conv_parameters_1.biases) free(weights->conv_parameters_1.biases);

    if (weights->conv_parameters_2.weights) {
        for (int i = 0; i < weights->conv_parameters_2.count; i++) {
            if (weights->conv_parameters_2.weights[i]) free_tensor(weights->conv_parameters_2.weights[i]);
        }
        free(weights->conv_parameters_2.weights);
    }
    if (weights->conv_parameters_2.biases) free(weights->conv_parameters_2.biases);

    if (weights->dense_parameters_1.weights) {
        for (int i = 0; i < weights->dense_parameters_1.count; i++) {
            if (weights->dense_parameters_1.weights[i]) free_tensor(weights->dense_parameters_1.weights[i]);
        }
        free(weights->dense_parameters_1.weights);
    }
    if (weights->dense_parameters_1.biases) free(weights->dense_parameters_1.biases);

    if (weights->dense_parameters_2.weights) {
        for (int i = 0; i < weights->dense_parameters_2.count; i++) {
            if (weights->dense_parameters_2.weights[i]) free_tensor(weights->dense_parameters_2.weights[i]);
        }
        free(weights->dense_parameters_2.weights);
    }
    if (weights->dense_parameters_2.biases) free(weights->dense_parameters_2.biases);

    if (weights->dense_parameters_3.weights) {
        for (int i = 0; i < weights->dense_parameters_3.count; i++) {
            if (weights->dense_parameters_3.weights[i]) free_tensor(weights->dense_parameters_3.weights[i]);
        }
        free(weights->dense_parameters_3.weights);
    }
    if (weights->dense_parameters_3.biases) free(weights->dense_parameters_3.biases);
}