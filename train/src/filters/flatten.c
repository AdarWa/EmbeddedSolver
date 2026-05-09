//
// Created by adarw on 4/16/26.
//

#include "flatten.h"

#include <stdlib.h>

tensor_t* flatten_tensor(tensor_t* input_tensor) {
    if (!input_tensor) return NULL;

    int size = 1;
    for (int i = 0; i < input_tensor->ndim; i++) {
        size *= input_tensor->shape[i];
    }

    tensor_t* output_tensor = allocate_tensor((int[]){size}, 1);
    if (!output_tensor) return NULL;

    // Allocate an array to track the index for the input tensor.
    int* element_idx = NULL;
    if (input_tensor->ndim > 0) {
        element_idx = (int*)calloc(input_tensor->ndim, sizeof(int));
    }

    for (int i = 0; i < size; i++) {
        double val = get_tensor_element(input_tensor, element_idx);

        int out_idx = i;
        set_tensor_element(output_tensor, &out_idx, val);

        for (int j = input_tensor->ndim - 1; j >= 0; j--) {
            element_idx[j]++;
            if (element_idx[j] < input_tensor->shape[j]) {
                break;
            }
            element_idx[j] = 0;
        }
    }

    if (element_idx) {
        free(element_idx);
    }

    return output_tensor;
}

tensor_t* unflatten_tensor_backward(const tensor_t* d_upstream, const tensor_t* original_input) {
    if (!d_upstream || !original_input) return NULL;

    tensor_t* d_input = allocate_tensor(original_input->shape, original_input->ndim);
    if (!d_input) return NULL;

    int size = 1;
    for (int i = 0; i < original_input->ndim; i++) {
        size *= original_input->shape[i];
    }

    int* element_idx = NULL;
    if (original_input->ndim > 0) {
        element_idx = (int*)calloc(original_input->ndim, sizeof(int));
    }

    for (int i = 0; i < size; i++) {
        // Get the gradient value from the 1D upstream tensor
        int upstream_idx = i;
        double grad_val = get_tensor_element(d_upstream, &upstream_idx);

        set_tensor_element(d_input, element_idx, grad_val);

        for (int j = original_input->ndim - 1; j >= 0; j--) {
            element_idx[j]++;
            if (element_idx[j] < original_input->shape[j]) {
                break;
            }
            element_idx[j] = 0;
        }
    }

    if (element_idx) free(element_idx);

    return d_input;
}
