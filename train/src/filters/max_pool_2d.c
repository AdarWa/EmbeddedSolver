//
// Created by adarw on 4/16/26.
//

#include "max_pool_2d.h"

#include <assert.h>
#include <math.h>

#include "../math/utils.h"


tensor_t* max_pool_2d_forward(tensor_t* input_tensor, int pool_height, int pool_width,
                             tensor_t** out_h_idx, tensor_t** out_w_idx) {
    int stride = pool_height;
    int in_height = input_tensor->shape[0];
    int in_width = input_tensor->shape[1];
    int in_channels = input_tensor->shape[2];

    int out_height = (in_height - pool_height) / stride + 1;
    int out_width = (in_width - pool_width) / stride + 1;

    tensor_t* output_tensor = allocate_tensor((int[]){out_height, out_width, in_channels}, 3);

    *out_h_idx = allocate_tensor((int[]){out_height, out_width, in_channels}, 3);
    *out_w_idx = allocate_tensor((int[]){out_height, out_width, in_channels}, 3);

    for (int c = 0; c < in_channels; c++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                int h_start = h * stride;
                int w_start = w * stride;

                double max_val = -INFINITY;
                int max_i = h_start;
                int max_j = w_start;

                for (int i = h_start; i < h_start + pool_height; i++) {
                    for (int j = w_start; j < w_start + pool_width; j++) {
                        double current_val = get_tensor_element(input_tensor, (int[]){i, j, c});
                        if (current_val > max_val) {
                            max_val = current_val;
                            max_i = i;
                            max_j = j;
                        }
                    }
                }

                set_tensor_element(output_tensor, (int[]){h, w, c}, max_val);
                // Store actual coordinates as doubles
                set_tensor_element(*out_h_idx, (int[]){h, w, c}, (double)max_i);
                set_tensor_element(*out_w_idx, (int[]){h, w, c}, (double)max_j);
            }
        }
    }
    return output_tensor;
}

tensor_t* max_pool_2d_backward(tensor_t* d_upstream, tensor_t* h_idx, tensor_t* w_idx,
                               int in_h, int in_w, int in_c) {

    tensor_t* d_input = allocate_tensor((int[]){in_h, in_w, in_c}, 3);

    int out_h = d_upstream->shape[0];
    int out_w = d_upstream->shape[1];

    for (int c = 0; c < in_c; c++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {

                double grad = get_tensor_element(d_upstream, (int[]){h, w, c});

                int orig_i = (int)get_tensor_element(h_idx, (int[]){h, w, c});
                int orig_j = (int)get_tensor_element(w_idx, (int[]){h, w, c});

                double current_val = get_tensor_element(d_input, (int[]){orig_i, orig_j, c});
                set_tensor_element(d_input, (int[]){orig_i, orig_j, c}, current_val + grad);
            }
        }
    }

    return d_input;
}
