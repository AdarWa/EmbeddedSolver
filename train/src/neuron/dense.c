//
// Created by adarw on 4/20/26.
//

#include "dense.h"

#include "logging/logging.h"
#include "math/utils.h"
#include "activation.h"

static const char* TAG = "neuron/dense";

tensor_t* dense_layer_forward_pass(tensor_t* input, int count, tensor_t** weights, double* biases, dense_activation_t activation) {

    if (!input || !weights || !biases) {
        LOG_E(TAG, "Passed NULL types to dense layer forward pass!");
        return NULL;
    }

    tensor_t* output = allocate_tensor((int[]){count}, 1);

    tensor_t* weight_matrix = *weights;

    int num_inputs = input->shape[0];

    for (int neuron = 0; neuron < count; neuron++) {
        double dot_product = 0.0;

        for (int i = 0; i < num_inputs; i++) {

            double x = get_tensor_element(input, (int[]){i});

            double w = get_tensor_element(weight_matrix, (int[]){i, neuron});

            dot_product += x * w;
        }

        double z = dot_product + biases[neuron];

        set_tensor_element(output, (int[]){neuron}, z);
    }

    if (activation == RELU) {
        for (int neuron = 0; neuron < count; neuron++) {
            set_tensor_element(output, (int[]){neuron},MAX(get_tensor_element(output, (int[]){neuron}), 0.0));
        }
        return output;
    }
    if (activation == SOFTMAX) {
        tensor_t* activated = softmax(output, count);
        free_tensor(output);
        return activated;
    }
    LOG_F(TAG, "Invalid activation type in dense layer %d", activation);
    return NULL;
}

tensor_t* dropout_forward_pass(const tensor_t* input, double drop_rate) {
    if (!input || !input->data) {
        LOG_E(TAG, "Invalid input passed to dropout.");
        return NULL;
    }

    tensor_t* output = allocate_tensor(input->shape, input->ndim);

    size_t total_elements = 1;
    for (int i = 0; i < input->ndim; i++) {
        total_elements *= input->shape[i];
    }

    double scale = 1.0 / (1.0 - drop_rate);

    for (size_t i = 0; i < total_elements; i++) {

        double rand_val = (double)rand() / RAND_MAX;

        if (rand_val < drop_rate) {
            output->data[i] = 0.0;
        } else {
            output->data[i] = input->data[i] * scale;
        }
    }

    return output;
}
