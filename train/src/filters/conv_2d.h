//
// Created by adarw on 4/10/26.
//

#pragma once

#include "../tensor/tensor.h"

tensor_t* conv2d_forward_pass(const tensor_t* input_tensor, tensor_t** weights, const double* biases, int num_filters, int kernel_size);