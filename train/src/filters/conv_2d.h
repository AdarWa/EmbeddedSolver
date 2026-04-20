//
// Created by adarw on 4/10/26.
//

#pragma once

#include "../tensor/tensor.h"

tensor_t* conv2d_forward_pass(const tensor_t* input_tensor, tensor_t** weights, const double* biases, int num_filters, int kernel_size);
tensor_t* conv2d_backward_pass(const tensor_t* d_out, const tensor_t* input,
                               const tensor_t* output_forward, tensor_t** weights,
                               tensor_t** d_weights, double* d_biases,
                               int num_filters, int kernel_size);