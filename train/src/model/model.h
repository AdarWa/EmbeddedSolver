//
// Created by adarw on 4/16/26.
//

#pragma once

#include "../tensor/tensor.h"

typedef enum {
    CONV_2D,
    MAX_POOL_2D,
    FLATTEN,
    DENSE,
    DROPOUT
} model_step_type;

typedef struct {
    int filters;
    int kernel_width;
    int kernel_height;
} conv_2d_params_t;

typedef struct {
    int pool_size;
} max_pool_2d_params_t;

typedef struct {
    int units;
} dense_params_t;

typedef struct {
    float rate;
} dropout_params_t;

typedef struct {
    model_step_type type;

    union {
        conv_2d_params_t     conv2d;
        max_pool_2d_params_t max_pool;
        dense_params_t       dense;
        dropout_params_t     dropout;
    };

} model_step_t;

tensor_t* process_model_step(tensor_t* input_tensor, model_step_t step);