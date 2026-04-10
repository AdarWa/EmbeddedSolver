//
// Created by adarw on 4/10/26.
//

#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../logging/logging.h"

static const char* TAG = "tensor";

tensor_t* allocate_tensor(const int* shape, int ndim) {
    tensor_t* tensor = malloc(sizeof(tensor_t));
    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->shape = malloc(ndim * sizeof(int));
    tensor->strides = malloc(ndim * sizeof(int));

    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape); free(tensor->strides); free(tensor);
        return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(int));

    // Precompute strides for row major order
    size_t total_elements = 1;
    int current_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = current_stride;
        current_stride *= shape[i];
    }
    total_elements = (size_t)current_stride;

    tensor->data = calloc(total_elements, sizeof(double));
    if (!tensor->data) {
        free(tensor->shape); free(tensor->strides); free(tensor);
        return NULL;
    }

    return tensor;
}

double get_tensor_element(const tensor_t* tensor, const int* element_idx) {
    if (!tensor || !element_idx) return NAN;

    int flat_idx = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        // Optional: Bounds checking
        if (element_idx[i] < 0 || element_idx[i] >= tensor->shape[i]) return NAN;

        flat_idx += element_idx[i] * tensor->strides[i];
    }
    return tensor->data[flat_idx];
}

void free_tensor(tensor_t* tensor) {
    if (!tensor) return;
    free(tensor->shape);
    free(tensor->strides);
    free(tensor->data);
    free(tensor);
}

void set_tensor_element(tensor_t* tensor, const int* element_idx, double value) {
    if (tensor == NULL || element_idx == NULL || tensor->data == NULL) {
        LOG_F(TAG, "Invalid arguments passed; set_tensor_element");
        return;
    }

    int flat_idx = 0;

    for (int i = 0; i < tensor->ndim; i++) {

        if (element_idx[i] < 0 || element_idx[i] >= tensor->shape[i]) {
            LOG_E(TAG, "Index %d out of bounds for dimension %d (size %d)",
                  element_idx[i], i, tensor->shape[i]);
            return;
        }

        flat_idx += element_idx[i] * tensor->strides[i];
    }

    tensor->data[flat_idx] = value;
}