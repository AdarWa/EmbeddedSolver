//
// Created by adarw on 4/20/26.
//

// test_dense.c
#include "unity.h"
#include <math.h>
#include <stdlib.h>

#include "neuron/dense.h"

static tensor_t* input_tensor = NULL;
static tensor_t* weight_tensor = NULL;
static tensor_t* output_tensor = NULL;
static double* biases = NULL;

void setUp(void) {
    input_tensor = NULL;
    weight_tensor = NULL;
    output_tensor = NULL;
    biases = NULL;
}

void tearDown(void) {
    if (input_tensor) free_tensor(input_tensor);
    if (weight_tensor) free_tensor(weight_tensor);
    if (output_tensor) free_tensor(output_tensor);
    if (biases) free(biases);
}

void test_dense_forward_pass_math(void) {
    input_tensor = allocate_tensor((int[]){3}, 1);
    input_tensor->data[0] = 1.0;
    input_tensor->data[1] = 2.0;
    input_tensor->data[2] = 3.0;

    weight_tensor = allocate_tensor((int[]){3, 2}, 2);
    weight_tensor->data[0] = 0.5;  // Input 0 -> Neuron 0
    weight_tensor->data[1] = 0.1;  // Input 0 -> Neuron 1
    weight_tensor->data[2] = -0.5; // Input 1 -> Neuron 0
    weight_tensor->data[3] = 0.2;  // Input 1 -> Neuron 1
    weight_tensor->data[4] = 1.0;  // Input 2 -> Neuron 0
    weight_tensor->data[5] = 0.3;  // Input 2 -> Neuron 1

    tensor_t* weights_array[] = {weight_tensor};

    biases = malloc(2 * sizeof(double));
    biases[0] = 0.5;
    biases[1] = -0.5;

    output_tensor = dense_layer_forward_pass(input_tensor, 2, weights_array, biases, RELU);

    TEST_ASSERT_NOT_NULL(output_tensor);
    TEST_ASSERT_EQUAL_INT(1, output_tensor->ndim);
    TEST_ASSERT_EQUAL_INT(2, output_tensor->shape[0]);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0, output_tensor->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.9, output_tensor->data[1]);
}

void test_dense_null_input_handling(void) {
    tensor_t* dummy_weight = allocate_tensor((int[]){1, 1}, 2);
    tensor_t* weights_array[] = {dummy_weight};
    double dummy_bias = 0.0;

    tensor_t* result = dense_layer_forward_pass(NULL, 1, weights_array, &dummy_bias, RELU);

    TEST_ASSERT_NULL(result);
    free_tensor(dummy_weight);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_dense_forward_pass_math);
    RUN_TEST(test_dense_null_input_handling);
    return UNITY_END();
}