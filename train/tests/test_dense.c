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

void test_dense_backward_basic_linear(void) {
    // 1. Setup Input (2 inputs)
    tensor_t* input = allocate_tensor((int[]){2}, 1);
    set_tensor_element(input, (int[]){0}, 1.0);
    set_tensor_element(input, (int[]){1}, 2.0);

    // 2. Setup Weights (2x1 matrix)
    tensor_t* weights = allocate_tensor((int[]){2, 1}, 2);
    set_tensor_element(weights, (int[]){0, 0}, 2.0);
    set_tensor_element(weights, (int[]){1, 0}, 3.0);

    // 3. Setup Forward Output (needed for potential activation checks)
    tensor_t* forward_out = allocate_tensor((int[]){1}, 1);
    set_tensor_element(forward_out, (int[]){0}, 8.0);

    // 4. Setup Upstream Gradient (1 neuron)
    tensor_t* d_out = allocate_tensor((int[]){1}, 1);
    set_tensor_element(d_out, (int[]){0}, 5.0);

    // 5. Setup Accumulators
    tensor_t* d_weights = allocate_tensor((int[]){2, 1}, 2);
    double d_biases[] = { 0.0 };

    // 6. Execute
    tensor_t* d_input = dense_layer_backward_pass(d_out, input, forward_out, weights,
                                                   d_weights, d_biases, 1, IDENTITY);

    // --- ASSERTIONS ---
    // Bias Gradient: dL/db = d_out = 5.0
    TEST_ASSERT_EQUAL_DOUBLE(5.0, d_biases[0]);

    // Weight Gradients: dL/dw = x * d_out
    // dw[0,0] = 1.0 * 5.0 = 5.0
    // dw[1,0] = 2.0 * 5.0 = 10.0
    TEST_ASSERT_EQUAL_DOUBLE(5.0,  get_tensor_element(d_weights, (int[]){0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(10.0, get_tensor_element(d_weights, (int[]){1, 0}));

    // Input Gradients: dL/dx = sum(w * d_out)
    // dx[0] = 2.0 * 5.0 = 10.0
    // dx[1] = 3.0 * 5.0 = 15.0
    TEST_ASSERT_EQUAL_DOUBLE(10.0, get_tensor_element(d_input, (int[]){0}));
    TEST_ASSERT_EQUAL_DOUBLE(15.0, get_tensor_element(d_input, (int[]){1}));

    free_tensor(input);
    free_tensor(weights);
    free_tensor(forward_out);
    free_tensor(d_out);
    free_tensor(d_weights);
    free_tensor(d_input);
}

// Test 2: Dense Backward with ReLU Activation
// Verifies that if forward output was 0, gradients are blocked.
void test_dense_backward_relu_blocking(void) {
    tensor_t* input = allocate_tensor((int[]){1}, 1);
    set_tensor_element(input, (int[]){0}, 10.0);

    tensor_t* weights = allocate_tensor((int[]){1, 1}, 2);
    set_tensor_element(weights, (int[]){0, 0}, 1.0);

    // Forward output is 0.0 (meaning it was clipped by ReLU)
    tensor_t* forward_out = allocate_tensor((int[]){1}, 1);
    set_tensor_element(forward_out, (int[]){0}, 0.0);

    tensor_t* d_out = allocate_tensor((int[]){1}, 1);
    set_tensor_element(d_out, (int[]){0}, 50.0);

    tensor_t* d_weights = allocate_tensor((int[]){1, 1}, 2);
    double d_biases[] = { 0.0 };

    tensor_t* d_input = dense_layer_backward_pass(d_out, input, forward_out, weights,
                                                   d_weights, d_biases, 1, RELU);

    // Since forward_out <= 0, all gradients should be 0.0
    TEST_ASSERT_EQUAL_DOUBLE(0.0, d_biases[0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(d_weights, (int[]){0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(d_input, (int[]){0}));

    free_tensor(input);
    free_tensor(weights);
    free_tensor(forward_out);
    free_tensor(d_out);
    free_tensor(d_weights);
    free_tensor(d_input);
}

// Test 3: Multiple Neurons (Output count > 1)
void test_dense_backward_multi_neuron(void) {
    // 1 input, 2 neurons
    tensor_t* input = allocate_tensor((int[]){1}, 1);
    set_tensor_element(input, (int[]){0}, 5.0);

    tensor_t* weights = allocate_tensor((int[]){1, 2}, 2);
    set_tensor_element(weights, (int[]){0, 0}, 2.0); // W for neuron 0
    set_tensor_element(weights, (int[]){0, 1}, 4.0); // W for neuron 1

    tensor_t* forward_out = allocate_tensor((int[]){2}, 1); // Not used in IDENTITY

    tensor_t* d_out = allocate_tensor((int[]){2}, 1);
    set_tensor_element(d_out, (int[]){0}, 10.0); // Gradient for neuron 0
    set_tensor_element(d_out, (int[]){1}, 20.0); // Gradient for neuron 1

    tensor_t* d_weights = allocate_tensor((int[]){1, 2}, 2);
    double d_biases[] = { 0.0, 0.0 };

    tensor_t* d_input = dense_layer_backward_pass(d_out, input, forward_out, weights,
                                                   d_weights, d_biases, 2, IDENTITY);

    // Input Gradient: (W0 * d_out0) + (W1 * d_out1)
    // dx = (2.0 * 10.0) + (4.0 * 20.0) = 20 + 80 = 100.0
    TEST_ASSERT_EQUAL_DOUBLE(100.0, get_tensor_element(d_input, (int[]){0}));

    free_tensor(input);
    free_tensor(weights);
    free_tensor(forward_out);
    free_tensor(d_out);
    free_tensor(d_weights);
    free_tensor(d_input);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_dense_forward_pass_math);
    RUN_TEST(test_dense_null_input_handling);
    RUN_TEST(test_dense_backward_relu_blocking);
    RUN_TEST(test_dense_backward_multi_neuron);
    return UNITY_END();
}