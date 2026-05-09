//
// Created by adarw on 4/16/26.
//

#pragma once

#include "../tensor/tensor.h"

tensor_t* max_pool_2d_forward(tensor_t* input_tensor, int pool_height, int pool_width,
                             tensor_t** out_h_idx, tensor_t** out_w_idx);
tensor_t* max_pool_2d_backward(tensor_t* d_upstream, tensor_t* h_idx, tensor_t* w_idx,
                               int in_h, int in_w, int in_c);