//
// Created by adarw on 4/20/26.
//

#pragma once
#include "tensor/tensor.h"

tensor_t* compute_softmax_cross_entropy_gradient(const tensor_t* predictions, int target_label);
