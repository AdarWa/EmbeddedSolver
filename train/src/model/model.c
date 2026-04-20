//
// Created by adarw on 4/17/26.
//

#include "model.h"

#include "backpropagation.h"
#include "batch.h"
#include "dataset/img_handler/img_handler.h"
#include "evaluation/evaluation.h"
#include "filters/conv_2d.h"
#include "filters/flatten.h"
#include "filters/max_pool_2d.h"
#include "logging/logging.h"
#include "neuron/dense.h"

static const char* TAG = "model";

static void zero_weights(weights_t* w) {
    tensor_zero(*w->conv_parameters_1.weights);
    for (int i = 0; i < w->conv_parameters_1.count; i++) {
        w->conv_parameters_2.biases[i] = 0;
    }

    tensor_zero(*w->conv_parameters_2.weights);
    for (int i = 0; i < w->conv_parameters_2.count; i++) {
        w->conv_parameters_2.biases[i] = 0;
    }

    tensor_zero(*w->dense_parameters_1.weights);
    for (int i = 0; i < w->dense_parameters_1.count; i++) {
        w->dense_parameters_1.biases[i] = 0;
    }

    tensor_zero(*w->dense_parameters_2.weights);
    for (int i = 0; i < w->dense_parameters_2.count; i++) {
        w->dense_parameters_2.biases[i] = 0;
    }

    tensor_zero(*w->dense_parameters_3.weights);
    for (int i = 0; i < w->dense_parameters_3.count; i++) {
        w->dense_parameters_3.biases[i] = 0;
    }
}

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
void train_model(model_train_config_t config, mnist_dataset_t* train, mnist_dataset_t* test) {
    train->count = 1000;
    weights_t weights = allocate_weights(&config);

    // Allocate a structure to hold accumulated gradients for the batch (init to 0)
    weights_t batch_gradients = allocate_weights(&config);

    uint32_t batch_cnt = get_batch_count(train, config.batch_size);
    batch_t* batches = get_batches(train, config.batch_size);

    LOG_I(TAG, "Allocated weights array, preparing for training");

    time_t ts = time(NULL);

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        time_t epoch_ts = time(NULL);
        for (int b = 0; b < batch_cnt; b++) {
            batch_t batch = batches[b];

            zero_weights(&batch_gradients);

            for (int i = 0; i < batch.cnt; i++) {
                tensor_t* input = batch.images[i].pixels;
                uint8_t target_label = batch.labels[i];
                LOG_I(TAG, "[Epoch %d/%d] Processing image %d/%d in batch %d/%d, target label %d", epoch, config.epochs, i, batch.cnt, b, batch_cnt, target_label);

                tensor_t* conv1_output = conv2d_forward_pass(input, weights.conv_parameters_1.weights, weights.conv_parameters_1.biases, config.filter_cnt_1, config.kernel_size);

                LOG_D(TAG, "[Epoch %d] Passed conv2d #1 forward pass", epoch);

                tensor_t *h_idx1 = NULL, *w_idx1 = NULL;
                tensor_t* max_pool1_output = max_pool_2d_forward(conv1_output, config.max_pool_size, config.max_pool_size, &h_idx1, &w_idx1);

                LOG_D(TAG, "[Epoch %d] Passed max pool2d #1 forward pass", epoch);

                // Layer 2: Conv -> Max Pool
                tensor_t* conv2_output = conv2d_forward_pass(max_pool1_output, weights.conv_parameters_2.weights, weights.conv_parameters_2.biases, config.filter_cnt_2, config.kernel_size);

                LOG_D(TAG, "[Epoch %d] Passed conv2d #2 forward pass", epoch);

                tensor_t *h_idx2 = NULL, *w_idx2 = NULL;
                tensor_t* max_pool2_output = max_pool_2d_forward(conv2_output, config.max_pool_size, config.max_pool_size, &h_idx2, &w_idx2);

                LOG_D(TAG, "[Epoch %d] Passed max pool2d #2 forward pass", epoch);

                // Flattening
                tensor_t* flattened = flatten_tensor(max_pool2_output);

                LOG_D(TAG, "[Epoch %d] Passed flatten forward pass", epoch);

                // Dense 1 -> Dropout 1
                tensor_t* dense1_output = dense_layer_forward_pass(flattened, config.dense_cnt_1, weights.dense_parameters_1.weights, weights.dense_parameters_1.biases, RELU);

                LOG_D(TAG, "[Epoch %d] Passed dense #1 forward pass(RELU)", epoch);

                tensor_t* dropout1_mask = NULL;
                tensor_t* dropout1_output = dropout_forward_pass(dense1_output, config.dropout_rate_1, &dropout1_mask);

                LOG_D(TAG, "[Epoch %d] Passed dropout #1 forward pass", epoch);

                // Dense 2 -> Dropout 2
                tensor_t* dense2_output = dense_layer_forward_pass(dropout1_output, config.dense_cnt_2, weights.dense_parameters_2.weights, weights.dense_parameters_2.biases, RELU);

                LOG_D(TAG, "[Epoch %d] Passed dense #2 forward pass(RELU)", epoch);

                tensor_t* dropout2_mask = NULL;
                tensor_t* dropout2_output = dropout_forward_pass(dense2_output, config.dropout_rate_2, &dropout2_mask);

                LOG_D(TAG, "[Epoch %d] Passed dropout #2 forward pass", epoch);

                // Dense 3
                tensor_t* final_predictions = dense_layer_forward_pass(dropout2_output, config.dense_cnt_3, weights.dense_parameters_3.weights, weights.dense_parameters_3.biases, SOFTMAX);

                LOG_D(TAG, "[Epoch %d] Passed dense #3 forward pass(SOFTMAX)", epoch);
                LOG_D(TAG, "[Epoch %d] Finished forward pass!", epoch);

                // Initial loss gradient:
                tensor_t* d_loss = compute_softmax_cross_entropy_gradient(final_predictions, target_label);

                // Dense 3 Backward
                tensor_t* d_dropout2 = dense_layer_backward_pass(d_loss, dropout2_output, final_predictions,
                                                                *weights.dense_parameters_3.weights, *batch_gradients.dense_parameters_3.weights,
                                                                batch_gradients.dense_parameters_3.biases, config.dense_cnt_3, SOFTMAX);

                // Dropout 2 Backward
                tensor_t* d_dense2 = dropout_backward_pass(d_dropout2, dropout2_mask, config.dropout_rate_2);

                // Dense 2 Backward
                tensor_t* d_dropout1 = dense_layer_backward_pass(d_dense2, dropout1_output, dense2_output,
                                                                *weights.dense_parameters_2.weights, *batch_gradients.dense_parameters_2.weights,
                                                                batch_gradients.dense_parameters_2.biases, config.dense_cnt_2, RELU);

                // Dropout 1 Backward
                tensor_t* d_dense1 = dropout_backward_pass(d_dropout1, dropout1_mask, config.dropout_rate_1);

                // Dense 1 Backward
                tensor_t* d_flattened = dense_layer_backward_pass(d_dense1, flattened, dense1_output,
                                                                 *weights.dense_parameters_1.weights, *batch_gradients.dense_parameters_1.weights,
                                                                 batch_gradients.dense_parameters_1.biases, config.dense_cnt_1, RELU);

                // Unflatten
                tensor_t* d_max_pool2 = unflatten_tensor_backward(d_flattened, max_pool2_output);

                // Max Pool 2 Backward
                tensor_t* d_conv2 = max_pool_2d_backward(d_max_pool2, h_idx2, w_idx2, conv2_output->shape[0], conv2_output->shape[1], conv2_output->shape[2]);

                // Conv 2 Backward
                tensor_t* d_max_pool1 = conv2d_backward_pass(d_conv2, max_pool1_output, conv2_output,
                                                            weights.conv_parameters_2.weights, batch_gradients.conv_parameters_2.weights,
                                                            batch_gradients.conv_parameters_2.biases, config.filter_cnt_2, config.kernel_size);

                // Max Pool 1 Backward
                tensor_t* d_conv1 = max_pool_2d_backward(d_max_pool1, h_idx1, w_idx1, conv1_output->shape[0], conv1_output->shape[1], conv1_output->shape[2]);

                // Conv 1 Backward
                tensor_t* d_input = conv2d_backward_pass(d_conv1, input, conv1_output,
                                                        weights.conv_parameters_1.weights, batch_gradients.conv_parameters_1.weights,
                                                        batch_gradients.conv_parameters_1.biases, config.filter_cnt_1, config.kernel_size);

                // AHHHHHHHHHHHHHHHHHHHHHHH
                free_tensor(conv1_output); free_tensor(max_pool1_output); free_tensor(h_idx1); free_tensor(w_idx1);
                free_tensor(conv2_output); free_tensor(max_pool2_output); free_tensor(h_idx2); free_tensor(w_idx2);
                free_tensor(flattened);
                free_tensor(dense1_output); free_tensor(dropout1_output); free_tensor(dropout1_mask);
                free_tensor(dense2_output); free_tensor(dropout2_output); free_tensor(dropout2_mask);
                free_tensor(final_predictions);

                free_tensor(d_loss); free_tensor(d_dropout2); free_tensor(d_dense2);
                free_tensor(d_dropout1); free_tensor(d_dense1); free_tensor(d_flattened);
                free_tensor(d_max_pool2); free_tensor(d_conv2); free_tensor(d_max_pool1);
                free_tensor(d_conv1); free_tensor(d_input);
            }
        }
        LOG_I(TAG, "Finished epoch %d/%d. Took %d seconds", epoch, config.epochs ,time(NULL) - epoch_ts);
    }

    LOG_I(TAG, "Finished Training! Ran through %d epoches. Took %d seconds.", config.epochs, time(NULL) - ts);

    save_weights(&weights);
    LOG_I(TAG, "Saved weights to file!");

    LOG_I(TAG, "Testing on a %d test dataset", test->count);
    LOG_I(TAG, "Model Accuracy: %f", evaluate_accuracy(config, &weights, test));

    free_weights(&weights);
    free_weights(&batch_gradients);
    free_batches(batches, batch_cnt);
}