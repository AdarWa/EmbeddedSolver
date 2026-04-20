//
// Created by adarw on 4/20/26.
//

#pragma once
#include "dataset/img_handler/img_handler.h"
#include "model/allocations.h"

double evaluate_accuracy(model_train_config_t config, weights_t* weights, mnist_dataset_t* dataset);
