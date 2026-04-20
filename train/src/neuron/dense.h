//
// Created by adarw on 4/20/26.
//

#pragma once
#include "tensor/tensor.h"

typedef enum {
    RELU,
    SOFTMAX,
    IDENTITY
} dense_activation_t;

tensor_t* dense_layer_forward_pass(tensor_t* input, int count, tensor_t** weights, double* biases, dense_activation_t activation);
tensor_t* dense_layer_backward_pass(tensor_t* d_out, tensor_t* input, tensor_t* forward_output,
                                    tensor_t* weights, tensor_t* d_weights, double* d_biases,
                                    int count, dense_activation_t activation);
tensor_t* dropout_forward_pass(const tensor_t* input, double drop_rate, tensor_t** out_mask);
tensor_t* dropout_backward_pass(const tensor_t* d_out, const tensor_t* mask, double drop_rate);
