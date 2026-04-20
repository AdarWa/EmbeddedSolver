//
// Created by adarw on 4/10/26.
//

#include "conv_2d.h"

#include <assert.h>

#include "math/utils.h"

tensor_t* conv2d_forward_pass(const tensor_t* input_tensor, tensor_t** weights, const double* biases, const int num_filters, const int kernel_size) {
    assert(input_tensor->ndim == 3);
    int in_height = input_tensor->shape[0];
    int in_width = input_tensor->shape[1];
    int in_channels = input_tensor->shape[2];

    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;

    tensor_t* output_tensor = allocate_tensor((int[]){out_height, out_width, num_filters}, 3);

    for (int f = 0; f < num_filters; f++) {
        tensor_t* current_filter = weights[f];
        double current_bias = biases[f];

        for (int y = 0; y < out_height; y++) {
            for (int x = 0; x < out_width; x++) {
                tensor_t* image_patch = tensor_extract_patch(input_tensor, y, x, kernel_size);
                double dot_product = 0;

                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        for (int c = 0; c < in_channels; c++) {
                            double pixel_val = get_tensor_element(image_patch, (int[]){i,j,c});
                            double weight_val = get_tensor_element(current_filter, (int[]){i,j,c});
                            dot_product += pixel_val * weight_val;
                        }
                    }
                }

                double z = dot_product + current_bias;

                double activated_value = MAX(0, z); // RELU

                set_tensor_element(output_tensor, (int[]){y,x,f}, activated_value);
            }
        }
    }
    return output_tensor;
}

tensor_t* conv2d_backward_pass(const tensor_t* d_out, const tensor_t* input,
                               const tensor_t* output_forward, tensor_t** weights,
                               tensor_t** d_weights, double* d_biases,
                               int num_filters, int kernel_size) {

    int in_h = input->shape[0];
    int in_w = input->shape[1];
    int in_c = input->shape[2];

    int out_h = d_out->shape[0];
    int out_w = d_out->shape[1];

    tensor_t* d_input = allocate_tensor((int[]){in_h, in_w, in_c}, 3);

    for (int f = 0; f < num_filters; f++) {
        tensor_t* current_filter = weights[f];
        tensor_t* current_d_filter = d_weights[f];

        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {

                double forward_val = get_tensor_element(output_forward, (int[]){y, x, f});
                if (forward_val <= 0) continue;

                double upstream_grad = get_tensor_element(d_out, (int[]){y, x, f});

                // d_bias = sum of upstream gradients
                d_biases[f] += upstream_grad;

                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        for (int c = 0; c < in_c; c++) {

                            double pixel_val = get_tensor_element(input, (int[]){y + i, x + j, c});
                            double weight_val = get_tensor_element(current_filter, (int[]){i, j, c});

                            // d_L/d_W = input * upstream_grad
                            double current_dw = get_tensor_element(current_d_filter, (int[]){i, j, c});
                            set_tensor_element(current_d_filter, (int[]){i, j, c}, current_dw + (pixel_val * upstream_grad));

                            // d_L/d_X = weight * upstream_grad
                            double current_dx = get_tensor_element(d_input, (int[]){y + i, x + j, c});
                            set_tensor_element(d_input, (int[]){y + i, x + j, c}, current_dx + (weight_val * upstream_grad));
                        }
                    }
                }
            }
        }
    }

    return d_input;
}
