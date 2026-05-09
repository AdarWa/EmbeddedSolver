//
// Created by adarw on 4/19/26.
//

// WTF is this shit
// Fucking hate handling the heap

#include "allocations.h"
#include "logging/logging.h"
#include <stdlib.h>

static const char* TAG = "model/allocations";

static int calculate_flattened_size(model_train_config_t* config, int input_width, int input_height) {
    int conv1_w = input_width - config->kernel_size + 1;
    int conv1_h = input_height - config->kernel_size + 1;

    int pool1_w = conv1_w / config->max_pool_size;
    int pool1_h = conv1_h / config->max_pool_size;

    int conv2_w = pool1_w - config->kernel_size + 1;
    int conv2_h = pool1_h - config->kernel_size + 1;

    int pool2_w = conv2_w / config->max_pool_size;
    int pool2_h = conv2_h / config->max_pool_size;

    return pool2_w * pool2_h * config->filter_cnt_2;
}

static tensor_t* randomize_weight_tensor(tensor_t* input) {
    long size = tensor_size(input);
    for (long i = 0; i < size; i++) {
        input->data[i] = ((double)rand() / (double)RAND_MAX);
    }
    return input;
}

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

    w.dense_parameters_1.weights = calloc(1, sizeof(tensor_t*));
    w.dense_parameters_1.biases = calloc(w.dense_parameters_1.count, sizeof(double));

    w.dense_parameters_2.weights = calloc(1, sizeof(tensor_t*));
    w.dense_parameters_2.biases = calloc(w.dense_parameters_2.count, sizeof(double));

    w.dense_parameters_3.weights = calloc(1, sizeof(tensor_t*));
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

    for (int i = 0; i < w.conv_parameters_1.count; i++) {
        w.conv_parameters_1.weights[i] = randomize_weight_tensor(allocate_tensor((int[]){config->kernel_size, config->kernel_size, config->channel_cnt}, 3));
        if (!w.conv_parameters_1.weights[i]) goto tensor_alloc_err;
    }

    for (int i = 0; i < w.conv_parameters_2.count; i++) {
        w.conv_parameters_2.weights[i] = randomize_weight_tensor(allocate_tensor((int[]){config->kernel_size, config->kernel_size, config->filter_cnt_1}, 3));
        if (!w.conv_parameters_2.weights[i]) goto tensor_alloc_err;
    }

    int flat_size = calculate_flattened_size(config, config->img_size, config->img_size);
    w.dense_parameters_1.weights[0] = randomize_weight_tensor(allocate_tensor((int[]){flat_size, config->dense_cnt_1}, 2));
    if (!w.dense_parameters_1.weights[0]) goto tensor_alloc_err;

    w.dense_parameters_2.weights[0] = randomize_weight_tensor(allocate_tensor((int[]){config->dense_cnt_1, config->dense_cnt_2}, 2));
    if (!w.dense_parameters_2.weights[0]) goto tensor_alloc_err;

    w.dense_parameters_3.weights[0] = randomize_weight_tensor(allocate_tensor((int[]){config->dense_cnt_2, config->dense_cnt_3}, 2));
    if (!w.dense_parameters_3.weights[0]) goto tensor_alloc_err;

    return w;

tensor_alloc_err:
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
        if (weights->dense_parameters_1.weights[0]) free_tensor(weights->dense_parameters_1.weights[0]);
        free(weights->dense_parameters_1.weights);
    }
    if (weights->dense_parameters_1.biases) free(weights->dense_parameters_1.biases);

    if (weights->dense_parameters_2.weights) {
        if (weights->dense_parameters_2.weights[0]) free_tensor(weights->dense_parameters_2.weights[0]);
        free(weights->dense_parameters_2.weights);
    }
    if (weights->dense_parameters_2.biases) free(weights->dense_parameters_2.biases);

    if (weights->dense_parameters_3.weights) {
        if (weights->dense_parameters_3.weights[0]) free_tensor(weights->dense_parameters_3.weights[0]);
        free(weights->dense_parameters_3.weights);
    }
    if (weights->dense_parameters_3.biases) free(weights->dense_parameters_3.biases);
}

void save_weights(weights_t* w) {
    FILE *file = fopen("weights.bin", "wb");

    if (file == NULL) {
        LOG_E(TAG, "Could not open file for writing");
        return;
    }

    size_t written = fwrite(w, sizeof(weights_t), 1, file);

    if (written != 1) {
        LOG_E(TAG, "Failed to write data to file");
    }

    fclose(file);
}

void load_weights(weights_t* w) {
    FILE *file = fopen("weights.bin", "rb");

    if (file == NULL) {
        LOG_E(TAG, "Could not open file for reading");
        return;
    }

    size_t read_count = fread(w, sizeof(weights_t), 1, file);

    if (read_count != 1) {
        LOG_E(TAG, "Failed to read data from file");
    }

    fclose(file);
}