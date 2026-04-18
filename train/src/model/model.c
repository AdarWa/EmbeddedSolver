//
// Created by adarw on 4/17/26.
//

#include "model.h"

#include <stdlib.h>

#include "dataset/img_handler/img_handler.h"
#include "logging/logging.h"
#include "tensor/tensor.h"

static const char* TAG = "model";

void train_model(model_train_config_t config, mnist_dataset_t* train) {
    tensor_t** conv_weights = malloc(config.filter_cnt * sizeof(tensor_t*));
    if (!conv_weights) {
        LOG_F(TAG, "Couldn't allocate array for conv 2d filter weights!");
        return;
    }
    double* conv_biases = malloc(config.filter_cnt * sizeof(double));
    if (!conv_biases) {
        free(conv_weights);
        LOG_F(TAG, "Couldn't allocate array for conv 2d filter biases!");
        return;
    }
    for (int i = 0; i < config.filter_cnt; i++) {
        conv_weights[i] = allocate_tensor((int[]){config.kernel_size, config.kernel_size, config.channel_cnt}, 3);

    }

    free(conv_weights);
    free(conv_biases);
}
