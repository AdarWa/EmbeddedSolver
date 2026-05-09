//
// Created by adarw on 4/20/26.
//

#include "activation.h"

#include <math.h>
#include <stddef.h>

tensor_t* softmax(const tensor_t* input, int size) {
    if (size <= 0) return NULL;

    tensor_t* output = allocate_tensor((int[]){size}, 1);

    double max_val = get_tensor_element(input, (int[]){0});
    for (int i = 1; i < size; i++) {
        double val = get_tensor_element(input, (int[]){i});
        if (val > max_val) {
            max_val = val;
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < size; i++) {
        int idx[] = {i};
        double val = exp(get_tensor_element(input, idx) - max_val);
        set_tensor_element(output, idx, val);
        sum_exp += val;
    }

    for (int i = 0; i < size; i++) {
        int idx[] = {i};
        double val = get_tensor_element(output, idx);
        set_tensor_element(output, idx, val / sum_exp);
    }
    return output;
}
