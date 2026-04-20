//
// Created by adarw on 4/16/26.
//

#pragma once

#include "../tensor/tensor.h"

tensor_t* flatten_tensor(tensor_t* input_tensor);
tensor_t* unflatten_tensor_backward(const tensor_t* d_upstream, const tensor_t* original_input);