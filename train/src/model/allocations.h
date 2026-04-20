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
    int filter_cnt_1;
    int filter_cnt_2;
    int kernel_size;
    int max_pool_size;
    int dense_cnt_1;
    int dense_cnt_2;
    int dense_cnt_3;
    double dropout_rate_1;
    double dropout_rate_2;
} model_train_config_t;

typedef struct {
    tensor_t** weights;
    double* biases;
    int count;
} cost_parameters_t;


typedef struct {
    cost_parameters_t conv_parameters_1;
    cost_parameters_t conv_parameters_2;
    cost_parameters_t dense_parameters_1;
    cost_parameters_t dense_parameters_2;
    cost_parameters_t dense_parameters_3;
} weights_t;

weights_t allocate_weights(model_train_config_t* config);
void free_weights(weights_t* weights);
void save_weights(weights_t* w);
void load_weights(weights_t* w);