//
// Created by adarw on 4/19/26.
//

#include "allocations.h"

#include "logging/logging.h"

static const char* TAG = "model/allocations";

weights_t allocate_weights(model_train_config_t* config) {
    tensor_t** conv_weights = malloc(config->filter_cnt * sizeof(tensor_t*));
    if (!conv_weights) {
        LOG_F(TAG, "Couldn't allocate array for conv 2d filter weights!");
        return (weights_t){0};
    }

    double* conv_biases = malloc(config->filter_cnt * sizeof(double));
    if (!conv_biases) {
        free(conv_weights);
        LOG_F(TAG, "Couldn't allocate array for conv 2d filter biases!");
        return (weights_t){0};
    }

    tensor_t** dense_weights = malloc(config->dense_cnt * sizeof(tensor_t*));
    if (!dense_weights) {
        free(conv_weights);
        free(conv_biases);
        LOG_F(TAG, "Couldn't allocate array for dense weights!");
        return (weights_t){0};
    }

    double* dense_biases = malloc(config->dense_cnt * sizeof(double));
    if (!dense_biases) {
        free(conv_weights);
        free(conv_biases);
        free(dense_weights);
        LOG_F(TAG, "Couldn't allocate array for dense biases!");
        return (weights_t){0};
    }

    for (int i = 0; i < config->filter_cnt; i++) {
        conv_weights[i] = allocate_tensor((int[]){config->kernel_size, config->kernel_size, config->channel_cnt}, 3);
    }

    for (int i = 0; i < config->dense_cnt; i++) {
        dense_weights[i] = allocate_tensor((int[]){config->dense_cnt, config->dense_cnt, config->channel_cnt}, 3);
    }

    return (weights_t){
        .conv_parameters = {
            .count = config->filter_cnt,
            .weights = conv_weights,
            .biases = conv_biases
        },
        .dense_parameters = {
            .count = config->dense_cnt,
            .weights = dense_weights,
            .biases = dense_biases
        }
    };
}

void free_weights(weights_t* weights) {
    if (!weights) return;

    if (weights->conv_parameters.weights) {
        for (int i = 0; i < weights->conv_parameters.count; i++) {
            if (weights->conv_parameters.weights[i]) {
                free_tensor(weights->conv_parameters.weights[i]);
            }
        }
        free(weights->conv_parameters.weights);
    }

    if (weights->conv_parameters.biases) {
        free(weights->conv_parameters.biases);
    }

    if (weights->dense_parameters.weights) {
        for (int i = 0; i < weights->dense_parameters.count; i++) {
            if (weights->dense_parameters.weights[i]) {
                free_tensor(weights->dense_parameters.weights[i]);
            }
        }
        free(weights->dense_parameters.weights);
    }

    if (weights->dense_parameters.biases) {
        free(weights->dense_parameters.biases);
    }
}