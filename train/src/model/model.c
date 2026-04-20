//
// Created by adarw on 4/17/26.
//

#include "model.h"

#include "batch.h"
#include "dataset/img_handler/img_handler.h"
#include "filters/conv_2d.h"
#include "filters/flatten.h"
#include "filters/max_pool_2d.h"
#include "logging/logging.h"
#include "neuron/dense.h"

static const char* TAG = "model";

/*
 * This function trains the model. The same in TensorFlow would be the following:
 *
 * model = Sequential()
 *
 * model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=INPUT_SHAPE))
 * model.add(MaxPool2D((2,2)))
 *
 * model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
 * model.add(MaxPool2D((2,2)))
 *
 * model.add(Flatten())
 *
 * model.add(Dense(128, activation='relu'))
 * model.add(Dropout(0.2))
 *
 * model.add(Dense(64, activation='relu'))
 * model.add(Dropout(0.2))
 *
 * model.add(Dense(10, activation='softmax'))
 **/
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
                tensor_t* conv1_output = conv2d_forward_pass(input, weights.conv_parameters_1.weights, weights.conv_parameters_1.biases,
                                    config.filter_cnt_1, config.kernel_size);
                tensor_t* max_pool1_output = max_pool_2d(conv1_output, config.max_pool_size, config.max_pool_size);
                free_tensor(conv1_output);
                tensor_t* conv2_output = conv2d_forward_pass(max_pool1_output, weights.conv_parameters_2.weights,
                                                             weights.conv_parameters_2.biases, config.filter_cnt_2,
                                                             config.kernel_size);
                free_tensor(max_pool1_output);
                tensor_t* max_pool2_output = max_pool_2d(conv2_output, config.max_pool_size, config.max_pool_size);
                free_tensor(conv2_output);
                tensor_t* flattened = flatten_tensor(max_pool2_output);
                free_tensor(max_pool2_output);
                tensor_t* dense1 = dense_layer_forward_pass(flattened, config.dense_cnt_1, weights.dense_parameters_1.weights, weights.dense_parameters_1.biases, RELU);
                free_tensor(flattened);
                tensor_t* dropout1 = dropout_forward_pass(dense1, config.dropout_rate_1);
                free_tensor(dense1);
                tensor_t* dense2 = dense_layer_forward_pass(dropout1, config.dense_cnt_2, weights.dense_parameters_2.weights, weights.dense_parameters_2.biases, RELU);
                free_tensor(dropout1);
                tensor_t* dropout2 = dropout_forward_pass(dense2, config.dropout_rate_2);
                free_tensor(dense2);

                tensor_t* final = dense_layer_forward_pass(dropout2, config.dense_cnt_3, weights.dense_parameters_3.weights, weights.dense_parameters_3.biases, SOFTMAX);


            }
        }
    }

    free_weights(&weights);
}
