//
// Created by adarw on 4/17/26.
//

#pragma once
#include "allocations.h"
#include "dataset/img_handler/img_handler.h"


void train_model(model_train_config_t config, mnist_dataset_t* train, mnist_dataset_t* test);
