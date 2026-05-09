//
// Created by adarw on 4/20/26.
//

#include "unity.h"
#include <stdlib.h>
#include <math.h>

#include "neuron/dense.h"

// Static pointers for easy cleanup
static tensor_t* input_tensor = NULL;
static tensor_t* output_tensor = NULL;
static tensor_t* mask_tensor = NULL;
static tensor_t* d_out_tensor = NULL;
static tensor_t* d_input_tensor = NULL;

void setUp(void) {
    input_tensor = NULL;
    output_tensor = NULL;
    mask_tensor = NULL;
    d_out_tensor = NULL;
    d_input_tensor = NULL;
    srand(42);
}

void tearDown(void) {
    if (input_tensor) free_tensor(input_tensor);
    if (output_tensor) free_tensor(output_tensor);
    if (mask_tensor) free_tensor(mask_tensor);
    if (d_out_tensor) free_tensor(d_out_tensor);
    if (d_input_tensor) free_tensor(d_input_tensor);
}

// --- UPDATED FORWARD TESTS ---

void test_dropout_zero_rate(void) {
    input_tensor = allocate_tensor((int[]){5}, 1);
    for (int i = 0; i < 5; i++) input_tensor->data[i] = 1.0;

    // Passing &mask_tensor to capture the dropout mask
    output_tensor = dropout_forward_pass(input_tensor, 0.0, &mask_tensor);

    TEST_ASSERT_NOT_NULL(output_tensor);
    TEST_ASSERT_NOT_NULL(mask_tensor);

    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0, output_tensor->data[i]);
        // With 0% drop rate, mask should be all 1s
        TEST_ASSERT_EQUAL_DOUBLE(1.0, mask_tensor->data[i]);
    }
}

void test_dropout_statistical_distribution(void) {
    int num_elements = 1000;
    double drop_rate = 0.5;
    double expected_scale = 1.0 / (1.0 - drop_rate);

    input_tensor = allocate_tensor((int[]){num_elements}, 1);
    for (int i = 0; i < num_elements; i++) input_tensor->data[i] = 1.0;

    output_tensor = dropout_forward_pass(input_tensor, drop_rate, &mask_tensor);

    int dropped_count = 0;
    for (int i = 0; i < num_elements; i++) {
        if (output_tensor->data[i] == 0.0) {
            dropped_count++;
            // Check that mask matches output zeros
            TEST_ASSERT_EQUAL_DOUBLE(0.0, mask_tensor->data[i]);
        } else {
            TEST_ASSERT_FLOAT_WITHIN(1e-5, expected_scale, output_tensor->data[i]);
            // Check that mask matches output active neurons
            TEST_ASSERT_EQUAL_DOUBLE(1.0, mask_tensor->data[i]);
        }
    }

    TEST_ASSERT_INT_WITHIN(50, 500, dropped_count);
}

void test_dropout_preserves_shape(void) {
    input_tensor = allocate_tensor((int[]){28, 28, 16}, 3);
    output_tensor = dropout_forward_pass(input_tensor, 0.2, &mask_tensor);

    TEST_ASSERT_EQUAL_INT(3, mask_tensor->ndim);
    TEST_ASSERT_EQUAL_INT(28, mask_tensor->shape[0]);
    TEST_ASSERT_EQUAL_INT(16, mask_tensor->shape[2]);
}

// --- NEW BACKWARD TEST ---

void test_dropout_backward_pass(void) {
    int num_elements = 10;
    double drop_rate = 0.5;
    double scale = 1.0 / (1.0 - drop_rate); // 2.0

    // 1. Setup Input and Run Forward to get a real mask
    input_tensor = allocate_tensor((int[]){num_elements}, 1);
    for (int i = 0; i < num_elements; i++) input_tensor->data[i] = 1.0;
    output_tensor = dropout_forward_pass(input_tensor, drop_rate, &mask_tensor);

    // 2. Setup a dummy upstream gradient (all 1.0s)
    d_out_tensor = allocate_tensor((int[]){num_elements}, 1);
    for (int i = 0; i < num_elements; i++) d_out_tensor->data[i] = 1.0;

    // 3. Run Backward Pass
    d_input_tensor = dropout_backward_pass(d_out_tensor, mask_tensor, drop_rate);

    // 4. Assert Gradient Logic
    TEST_ASSERT_NOT_NULL(d_input_tensor);
    for (int i = 0; i < num_elements; i++) {
        if (mask_tensor->data[i] == 0.0) {
            // If neuron was dropped, gradient must be 0
            TEST_ASSERT_EQUAL_DOUBLE(0.0, d_input_tensor->data[i]);
        } else {
            // If neuron was active, gradient must be upstream * scale
            TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0 * scale, d_input_tensor->data[i]);
        }
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_dropout_zero_rate);
    RUN_TEST(test_dropout_statistical_distribution);
    RUN_TEST(test_dropout_preserves_shape);
    RUN_TEST(test_dropout_backward_pass);
    return UNITY_END();
}