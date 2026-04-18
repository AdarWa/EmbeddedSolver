//
// Created by adarw on 4/19/26.
//

#pragma once

#include "tensor/tensor.h"

typedef struct {
    int epochs;
    int batch_size;
    int channel_cnt;
    int img_size;
    int filter_cnt;
    int kernel_size;
    int dense_cnt;
} model_train_config_t;

typedef struct {
    tensor_t** weights;
    double* biases;
    int count;
} cost_parameters_t;


typedef struct {
    cost_parameters_t conv_parameters;
    cost_parameters_t dense_parameters;
} weights_t;

weights_t allocate_weights(model_train_config_t* config);
void free_weights(weights_t* weights);