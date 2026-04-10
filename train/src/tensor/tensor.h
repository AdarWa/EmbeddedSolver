//
// Created by adarw on 4/10/26.
//

#pragma once

typedef struct {
    double* data;
    int* shape;
    int* strides;
    int ndim;
} tensor_t;

tensor_t* allocate_tensor(const int* shape, int ndim);
void free_tensor(tensor_t* tensor);
double get_tensor_element(const tensor_t* tensor, const int* element_idx);
void set_tensor_element(tensor_t* tensor, const int* element_idx, double value);