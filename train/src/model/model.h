//
// Created by adarw on 4/17/26.
//

#pragma once
#include "dataset/img_handler/img_handler.h"

typedef struct {
    int epochs;
    int batch_size;
    int channel_cnt;
    int img_size;
    int filter_cnt;
    int kernel_size;
} model_train_config_t;

void train_model(model_train_config_t config, mnist_dataset_t* train);
