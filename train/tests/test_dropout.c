//
// Created by adarw on 4/20/26.
//

#include "unity.h"
#include <stdlib.h>

#include "neuron/dense.h"

static tensor_t* input_tensor = NULL;
static tensor_t* output_tensor = NULL;

void setUp(void) {
    input_tensor = NULL;
    output_tensor = NULL;
    srand(42); // NOLINT(*-msc51-cpp)
}

void tearDown(void) {
    if (input_tensor) free_tensor(input_tensor);
    if (output_tensor) free_tensor(output_tensor);
}

void test_dropout_zero_rate(void) {
    input_tensor = allocate_tensor((int[]){5}, 1);
    for (int i = 0; i < 5; i++) input_tensor->data[i] = 1.0;

    output_tensor = dropout_forward_pass(input_tensor, 0.0);

    TEST_ASSERT_NOT_NULL(output_tensor);
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0, output_tensor->data[i]);
    }
}

void test_dropout_statistical_distribution(void) {
    int num_elements = 1000;
    double drop_rate = 0.5;
    double expected_scale = 1.0 / (1.0 - drop_rate);

    input_tensor = allocate_tensor((int[]){num_elements}, 1);
    for (int i = 0; i < num_elements; i++) {
        input_tensor->data[i] = 1.0;
    }

    output_tensor = dropout_forward_pass(input_tensor, drop_rate);
    TEST_ASSERT_NOT_NULL(output_tensor);

    int dropped_count = 0;
    int scaled_count = 0;

    for (int i = 0; i < num_elements; i++) {
        if (output_tensor->data[i] == 0.0) {
            dropped_count++;
        } else {
            TEST_ASSERT_FLOAT_WITHIN(1e-5, expected_scale, output_tensor->data[i]);
            scaled_count++;
        }
    }

    TEST_ASSERT_INT_WITHIN(50, 500, dropped_count);
    TEST_ASSERT_EQUAL_INT(num_elements, dropped_count + scaled_count);
}

void test_dropout_preserves_shape(void) {
    input_tensor = allocate_tensor((int[]){28, 28, 16}, 3);

    output_tensor = dropout_forward_pass(input_tensor, 0.2);

    TEST_ASSERT_NOT_NULL(output_tensor);
    TEST_ASSERT_EQUAL_INT(3, output_tensor->ndim);
    TEST_ASSERT_EQUAL_INT(28, output_tensor->shape[0]);
    TEST_ASSERT_EQUAL_INT(28, output_tensor->shape[1]);
    TEST_ASSERT_EQUAL_INT(16, output_tensor->shape[2]);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_dropout_zero_rate);
    RUN_TEST(test_dropout_statistical_distribution);
    RUN_TEST(test_dropout_preserves_shape);
    return UNITY_END();
}