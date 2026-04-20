//
// Created by adarw on 4/20/26.
//

#pragma once
#include "tensor/tensor.h"

typedef enum {
    RELU,
    SOFTMAX
} dense_activation_t;

tensor_t* dense_layer_forward_pass(tensor_t* input, int count, tensor_t** weights, double* biases, dense_activation_t activation);
tensor_t* dropout_forward_pass(const tensor_t* input, double drop_rate);
