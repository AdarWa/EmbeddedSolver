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

tensor_t* dense_layer_backward_pass(tensor_t* d_out, tensor_t* input, tensor_t* forward_output,
                                    tensor_t* weights, tensor_t* d_weights, double* d_biases,
                                    int count, dense_activation_t activation) {
    if (!d_out || !input || !forward_output || !weights) return NULL;

    int num_inputs = input->shape[0];

    tensor_t* d_input = allocate_tensor((int[]){num_inputs}, 1);

    double* dZ = (double*)calloc(count, sizeof(double));

    for (int j = 0; j < count; j++) {
        double grad_a = get_tensor_element(d_out, (int[]){j});

        if (activation == RELU) {
            // RELU Derivative: 1 if output was > 0, else 0
            double a = get_tensor_element(forward_output, (int[]){j});
            dZ[j] = (a > 0.0) ? grad_a : 0.0;
        }
        else if (activation == SOFTMAX || activation == IDENTITY) {
            dZ[j] = grad_a;
        }else {
            LOG_F(TAG, "Invalid activation type in dense layer backpropagation");
        }
    }

    // Calculate Weight, Bias, and Input Gradients
    for (int j = 0; j < count; j++) {

        // dL/db = dZ
        d_biases[j] += dZ[j];

        for (int i = 0; i < num_inputs; i++) {
            double x = get_tensor_element(input, (int[]){i});
            double w = get_tensor_element(weights, (int[]){i, j});

            // dL/dw = x * dZ
            // d_weights[num_inputs x count]
            double current_dw = get_tensor_element(d_weights, (int[]){i, j});
            set_tensor_element(d_weights, (int[]){i, j}, current_dw + (x * dZ[j]));

            // dL/dx = sum(w * dZ)
            double current_dx = get_tensor_element(d_input, (int[]){i});
            set_tensor_element(d_input, (int[]){i}, current_dx + (w * dZ[j]));
        }
    }

    free(dZ);
    return d_input;
}

tensor_t* dropout_forward_pass(const tensor_t* input, double drop_rate, tensor_t** out_mask) {
    tensor_t* output = allocate_tensor(input->shape, input->ndim);
    *out_mask = allocate_tensor(input->shape, input->ndim); // Store the mask

    size_t total_elements = 1;
    for (int i = 0; i < input->ndim; i++) total_elements *= input->shape[i];

    double scale = 1.0 / (1.0 - drop_rate);

    for (size_t i = 0; i < total_elements; i++) {
        double rand_val = (double)rand() / RAND_MAX;

        if (rand_val < drop_rate) {
            output->data[i] = 0.0;
            (*out_mask)->data[i] = 0.0;
        } else {
            output->data[i] = input->data[i] * scale;
            (*out_mask)->data[i] = 1.0;
        }
    }
    return output;
}

tensor_t* dropout_backward_pass(const tensor_t* d_out, const tensor_t* mask, double drop_rate) {
    if (!d_out || !mask) return NULL;

    tensor_t* d_input = allocate_tensor(d_out->shape, d_out->ndim);

    size_t total_elements = 1;
    for (int i = 0; i < d_out->ndim; i++) {
        total_elements *= d_out->shape[i];
    }

    // The derivative of Inverted Dropout is the mask scaled by 1/(1-p)
    double scale = 1.0 / (1.0 - drop_rate);

    for (size_t i = 0; i < total_elements; i++) {
        // If mask was 0, gradient is 0
        // If mask was 1, gradient is d_out * scale
        if (mask->data[i] == 0.0) {
            d_input->data[i] = 0.0;
        } else {
            d_input->data[i] = d_out->data[i] * scale;
        }
    }

    return d_input;
}