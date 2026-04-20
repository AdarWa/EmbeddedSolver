//
// Created by adarw on 4/20/26.
//

#include "evaluation.h"

#include <stddef.h>

#include "filters/conv_2d.h"
#include "filters/flatten.h"
#include "filters/max_pool_2d.h"
#include "neuron/dense.h"

double evaluate_accuracy(model_train_config_t config, weights_t* weights, mnist_dataset_t* dataset) {
    int correct_predictions = 0;

    for (int i = 0; i < dataset->count; i++) {
        tensor_t* input = dataset->images[i].pixels;
        uint8_t target_label = dataset->labels[i];

        tensor_t* conv1_output = conv2d_forward_pass(input, weights->conv_parameters_1.weights, weights->conv_parameters_1.biases, config.filter_cnt_1, config.kernel_size);
        tensor_t *h_idx1 = NULL, *w_idx1 = NULL;
        tensor_t* max_pool1_output = max_pool_2d_forward(conv1_output, config.max_pool_size, config.max_pool_size, &h_idx1, &w_idx1);

        tensor_t* conv2_output = conv2d_forward_pass(max_pool1_output, weights->conv_parameters_2.weights, weights->conv_parameters_2.biases, config.filter_cnt_2, config.kernel_size);
        tensor_t *h_idx2 = NULL, *w_idx2 = NULL;
        tensor_t* max_pool2_output = max_pool_2d_forward(conv2_output, config.max_pool_size, config.max_pool_size, &h_idx2, &w_idx2);

        tensor_t* flattened = flatten_tensor(max_pool2_output);

        tensor_t* dense1_output = dense_layer_forward_pass(flattened, config.dense_cnt_1, weights->dense_parameters_1.weights, weights->dense_parameters_1.biases, RELU);

        tensor_t* dense2_output = dense_layer_forward_pass(dense1_output, config.dense_cnt_2, weights->dense_parameters_2.weights, weights->dense_parameters_2.biases, RELU);

        tensor_t* final_predictions = dense_layer_forward_pass(dense2_output, config.dense_cnt_3, weights->dense_parameters_3.weights, weights->dense_parameters_3.biases, SOFTMAX);

        int predicted_label = 0;
        double max_probability = get_tensor_element(final_predictions, (int[]){0});

        for (int j = 1; j < 10; j++) {
            if (final_predictions->data[j] > max_probability) {
                max_probability = get_tensor_element(final_predictions, (int[]){i});
                predicted_label = j;
            }
        }

        if (predicted_label == target_label) {
            correct_predictions++;
        }

        free_tensor(conv1_output); free_tensor(max_pool1_output); free_tensor(h_idx1); free_tensor(w_idx1);
        free_tensor(conv2_output); free_tensor(max_pool2_output); free_tensor(h_idx2); free_tensor(w_idx2);
        free_tensor(flattened);
        free_tensor(dense1_output);
        free_tensor(dense2_output);
        free_tensor(final_predictions);
    }

    return ((double)correct_predictions / dataset->count) * 100.0;
}
