//
// Created by adarw on 4/10/26.
//

#include "conv_2d.h"

#include <assert.h>

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
