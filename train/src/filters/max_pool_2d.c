//
// Created by adarw on 4/16/26.
//

#include "max_pool_2d.h"

#include <assert.h>
#include <math.h>

#include "../math/utils.h"


tensor_t* max_pool_2d(tensor_t* input_tensor, int pool_height, int pool_width) {
    assert(input_tensor->ndim == 3);
    assert(pool_height == pool_width); // for simplicity
    int stride = pool_height;
    int in_height = input_tensor->shape[0];
    int in_width = input_tensor->shape[1];
    int in_channels = input_tensor->shape[2];

    int out_height = (in_height - pool_height) / stride + 1;
    int out_width = (in_width - pool_width) / stride + 1;

    tensor_t* output_tensor = allocate_tensor((int[]){out_height, out_width, in_channels}, 3);

    for (int c = 0; c < in_channels; c++) {
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w++) {
                int h_start = h*stride;
                int h_end = h_start + pool_height;
                int w_start = w*stride;
                int w_end = w_start + pool_width;

                double max = NAN;
                for (int i = h_start; i < h_end; i++) {
                    for (int j = w_start; j < w_end; j++) {
                        max = MAX(max, get_tensor_element(input_tensor, (int[]) {i, j, c}));
                    }
                }
                assert(!isnan(max));
                set_tensor_element(output_tensor, (int[]){h,w,c}, max);
            }
        }
    }

    return output_tensor;
}
