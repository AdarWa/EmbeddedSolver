//
// Created by adarw on 4/16/26.
//

#include <stdio.h>
#include "unity.h"
#include "../src/tensor/tensor.h"
#include "../src/filters/flatten.h"


void setUp(void) {}
void tearDown(void) {}

void test_flatten_null_tensor(void) {
    tensor_t* result = flatten_tensor(NULL);
    TEST_ASSERT_NULL(result);
}

void test_flatten_1d_tensor(void) {
    int shape[] = {5};
    tensor_t* t1 = allocate_tensor(shape, 1);

    for (int i = 0; i < 5; i++) t1->data[i] = (double)(i + 1);

    tensor_t* flat = flatten_tensor(t1);

    TEST_ASSERT_NOT_NULL(flat);
    TEST_ASSERT_EQUAL_INT(1, flat->ndim);
    TEST_ASSERT_EQUAL_INT(5, flat->shape[0]);

    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(t1->data[i], flat->data[i]);
    }

    free_tensor(t1);
    free_tensor(flat);
}

void test_flatten_2d_tensor(void) {
    int shape[] = {2, 3};
    tensor_t* t2 = allocate_tensor(shape, 2);

    // Fill 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
    double expected[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (int i = 0; i < 6; i++) {
        t2->data[i] = expected[i];
    }

    tensor_t* flat = flatten_tensor(t2);

    TEST_ASSERT_NOT_NULL(flat);
    TEST_ASSERT_EQUAL_INT(1, flat->ndim);
    TEST_ASSERT_EQUAL_INT(6, flat->shape[0]);

    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(expected[i], flat->data[i]);
    }

    free_tensor(t2);
    free_tensor(flat);
}

void test_flatten_3d_tensor(void) {
    int shape[] = {2, 2, 2};
    tensor_t* t3 = allocate_tensor(shape, 3);

    // Fill 2x2x2 tensor (size 8)
    for (int i = 0; i < 8; i++) {
        t3->data[i] = (double)(i * 2);
    }

    tensor_t* flat = flatten_tensor(t3);

    TEST_ASSERT_NOT_NULL(flat);
    TEST_ASSERT_EQUAL_INT(1, flat->ndim);
    TEST_ASSERT_EQUAL_INT(8, flat->shape[0]);

    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_EQUAL_DOUBLE((double)(i * 2), flat->data[i]);
    }

    free_tensor(t3);
    free_tensor(flat);
}

void test_flatten_backward_pass(void) {
    // 1. Setup Mock Original Input (2x2x2)
    int shape[] = {2, 2, 2};
    tensor_t* original_input = allocate_tensor(shape, 3);

    // 2. Setup Upstream Gradient (1D vector of length 8)
    // Let's say the gradient coming from the Dense layer is [1, 2, 3, 4, 5, 6, 7, 8]
    tensor_t* d_upstream = allocate_tensor((int[]){8}, 1);
    for (int i = 0; i < 8; i++) {
        int idx = i;
        set_tensor_element(d_upstream, &idx, (double)(i + 1));
    }

    // 3. Execute Backward Pass
    tensor_t* d_input = unflatten_tensor_backward(d_upstream, original_input);

    // 4. Assertions
    TEST_ASSERT_NOT_NULL(d_input);
    TEST_ASSERT_EQUAL_INT(3, d_input->ndim);
    TEST_ASSERT_EQUAL_INT(2, d_input->shape[0]);
    TEST_ASSERT_EQUAL_INT(2, d_input->shape[2]);

    // Check specific mapping (Assuming H-W-C order from your iterator)
    // index 0,0,0 should be 1.0
    TEST_ASSERT_EQUAL_DOUBLE(1.0, get_tensor_element(d_input, (int[]){0, 0, 0}));
    // index 1,1,1 should be 8.0
    TEST_ASSERT_EQUAL_DOUBLE(8.0, get_tensor_element(d_input, (int[]){1, 1, 1}));

    // Cleanup
    free_tensor(original_input);
    free_tensor(d_upstream);
    free_tensor(d_input);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_flatten_null_tensor);
    RUN_TEST(test_flatten_1d_tensor);
    RUN_TEST(test_flatten_2d_tensor);
    RUN_TEST(test_flatten_3d_tensor);
    RUN_TEST(test_flatten_backward_pass);

    return UNITY_END();
}