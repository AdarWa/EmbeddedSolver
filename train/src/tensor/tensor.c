//
// Created by adarw on 4/10/26.
//

#include "tensor.h"

#include <assert.h>
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

tensor_t* tensor_extract_patch(const tensor_t* src, int y, int x, int n) {
    if (!src || src->ndim != 3) {
        LOG_E(TAG, "extract_patch requires a 3D tensor.");
        return NULL;
    }

    if (y < 0 || x < 0 || y + n > src->shape[0] || x + n > src->shape[1]) {
        LOG_E(TAG, "Patch indices out of bounds.");
        return NULL;
    }

    int channels = src->shape[2];
    int patch_shape[] = {n, n, channels};
    tensor_t* patch = allocate_tensor(patch_shape, 3);
    if (!patch) return NULL;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < channels; k++) {
                int src_idx[] = {y + i, x + j, k};
                int dst_idx[] = {i, j, k};

                double val = get_tensor_element(src, src_idx);
                set_tensor_element(patch, dst_idx, val);
            }
        }
    }

    return patch;
}

double tensor_dot_product(const tensor_t* a, const tensor_t* b) {
    if (!a || !b || !a->data || !b->data) {
        LOG_E(TAG, "NULL tensor passed to dot product.");
        return NAN;
    }

    size_t size_a = 1;
    for (int i = 0; i < a->ndim; i++) size_a *= a->shape[i];

    size_t size_b = 1;
    for (int i = 0; i < b->ndim; i++) size_b *= b->shape[i];

    if (size_a != size_b) {
        LOG_E(TAG, "Tensor sizes do not match for dot product: %zu vs %zu", size_a, size_b);
        return NAN;
    }

    double result = 0.0;
    for (size_t i = 0; i < size_a; i++) {
        result += a->data[i] * b->data[i];
    }

    return result;
}

tensor_t* tensor_transpose(const tensor_t* input) {
    assert(input->ndim == 3);
    int m = input->shape[0];
    int n = input->shape[1];
    tensor_t* output = allocate_tensor((int[]){n,m,input->shape[2]},3);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            double val = get_tensor_element(input, (int[]){i,j,0});
            set_tensor_element(output, (int[]){j,i,0}, val);
        }
    }
    return output;
}

void tensor_zero(tensor_t* input) {
    if (!input || !input->data) {
        return;
    }
    long current_stride = 1;
    for (int i = input->ndim - 1; i >= 0; i--) {
        current_stride *= input->shape[i];
    }
    memset(input->data, 0, current_stride * sizeof(double));
}
