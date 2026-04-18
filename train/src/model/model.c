//
// Created by adarw on 4/17/26.
//

#include "model.h"

#include "batch.h"
#include "dataset/img_handler/img_handler.h"
#include "filters/conv_2d.h"
#include "logging/logging.h"

static const char* TAG = "model";

void train_model(model_train_config_t config, mnist_dataset_t* train) {
    weights_t weights = allocate_weights(&config);

    uint32_t batch_cnt = get_batch_count(train, config.batch_size);
    batch_t* batches = get_batches(train, config.batch_size);

    LOG_I(TAG, "Allocated weights array, preparing for training");

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        for (int b = 0; b < batch_cnt; b++) {
            batch_t batch = batches[b];
            for (int i = 0; i < batch.cnt; i++) {
                tensor_t* input = batch.images[i].pixels;
                tensor_t* conv_output = conv2d_forward_pass(input, weights.conv_parameters.weights, weights.conv_parameters.biases,
                                    config.filter_cnt, config.kernel_size);
                free_tensor(input);


            }
        }
    }

    free_weights(&weights);
}
