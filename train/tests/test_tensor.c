//
// Created by adarw on 4/10/26.
//

#include "unity.h"
#include "../src/tensor/tensor.h"
#include <math.h>
#include <stdlib.h>
#define TAG "TEST"

void setUp(void) {
    // Set up before each test if needed
}

void tearDown(void) {
    // Clean up after each test if needed
}


void test_tensor_allocation_success(void) {
    int shape[] = {2, 3, 4}; // 2x3x4 tensor
    int ndim = 3;
    tensor_t* t = allocate_tensor(shape, ndim);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_INT(ndim, t->ndim);

    // Check shapes
    for(int i = 0; i < ndim; i++) {
        TEST_ASSERT_EQUAL_INT(shape[i], t->shape[i]);
    }

    // Check Row-Major Strides: {12, 4, 1}
    TEST_ASSERT_EQUAL_INT(1, t->strides[2]);
    TEST_ASSERT_EQUAL_INT(4, t->strides[1]);
    TEST_ASSERT_EQUAL_INT(12, t->strides[0]);

    // Check data is zero-initialized (calloc)
    TEST_ASSERT_EQUAL_DOUBLE(0.0, t->data[0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, t->data[23]);

    free_tensor(t);
}

// --- Accessor Tests ---

void test_tensor_get_set_element(void) {
    int shape[] = {2, 2};
    tensor_t* t = allocate_tensor(shape, 2);

    int idx[] = {1, 0};
    set_tensor_element(t, idx, 42.5);

    TEST_ASSERT_EQUAL_DOUBLE(42.5, get_tensor_element(t, idx));

    // Verify internal flat mapping: idx {1, 0} in 2x2 should be flat index 2
    TEST_ASSERT_EQUAL_DOUBLE(42.5, t->data[2]);

    free_tensor(t);
}

void test_tensor_bounds_checking(void) {
    int shape[] = {2, 2};
    tensor_t* t = allocate_tensor(shape, 2);

    int bad_idx[] = {2, 0}; // Out of bounds

    // set_tensor_element handles bounds via LOG_E internally,
    // get_tensor_element returns NAN
    TEST_ASSERT_TRUE(isnan(get_tensor_element(t, bad_idx)));

    free_tensor(t);
}

// --- Logic Tests ---

void test_tensor_dot_product(void) {
    int shape[] = {3};
    tensor_t* a = allocate_tensor(shape, 1);
    tensor_t* b = allocate_tensor(shape, 1);

    // a = [1, 2, 3], b = [4, 5, 6]
    // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    int idx0[] = {0}, idx1[] = {1}, idx2[] = {2};
    set_tensor_element(a, idx0, 1.0); set_tensor_element(a, idx1, 2.0); set_tensor_element(a, idx2, 3.0);
    set_tensor_element(b, idx0, 4.0); set_tensor_element(b, idx1, 5.0); set_tensor_element(b, idx2, 6.0);

    double result = tensor_dot_product(a, b);
    TEST_ASSERT_EQUAL_DOUBLE(32.0, result);

    free_tensor(a);
    free_tensor(b);
}

void test_tensor_dot_product_mismatch(void) {
    int shape_a[] = {3};
    int shape_b[] = {4};
    tensor_t* a = allocate_tensor(shape_a, 1);
    tensor_t* b = allocate_tensor(shape_b, 1);

    TEST_ASSERT_TRUE(isnan(tensor_dot_product(a, b)));

    free_tensor(a);
    free_tensor(b);
}

void test_tensor_extract_patch(void) {
    // Create 4x4x1 tensor (like a grayscale image)
    int shape[] = {4, 4, 1};
    tensor_t* src = allocate_tensor(shape, 3);

    // Fill with values: [y][x][0] = y * 10 + x
    for(int y=0; y<4; y++) {
        for(int x=0; x<4; x++) {
            int idx[] = {y, x, 0};
            set_tensor_element(src, idx, (double)(y * 10 + x));
        }
    }

    // Extract 2x2 patch from y=1, x=1
    // Expected values: [11, 12], [21, 22]
    tensor_t* patch = tensor_extract_patch(src, 1, 1, 2);

    TEST_ASSERT_NOT_NULL(patch);
    TEST_ASSERT_EQUAL_INT(2, patch->shape[0]);
    TEST_ASSERT_EQUAL_INT(2, patch->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, patch->shape[2]);

    int p_idx00[] = {0, 0, 0}; TEST_ASSERT_EQUAL_DOUBLE(11.0, get_tensor_element(patch, p_idx00));
    int p_idx01[] = {0, 1, 0}; TEST_ASSERT_EQUAL_DOUBLE(12.0, get_tensor_element(patch, p_idx01));
    int p_idx10[] = {1, 0, 0}; TEST_ASSERT_EQUAL_DOUBLE(21.0, get_tensor_element(patch, p_idx10));
    int p_idx11[] = {1, 1, 0}; TEST_ASSERT_EQUAL_DOUBLE(22.0, get_tensor_element(patch, p_idx11));

    free_tensor(src);
    free_tensor(patch);
}

void test_tensor_extract_patch_invalid(void) {
    int shape[] = {3, 3, 1};
    tensor_t* src = allocate_tensor(shape, 3);

    // Extraction exceeds bounds (y+n > shape[0])
    tensor_t* patch = tensor_extract_patch(src, 2, 2, 2);
    TEST_ASSERT_NULL(patch);

    free_tensor(src);
}

void test_tensor_zero_basic(void) {
    int shape[] = {2, 2};
    tensor_t* t = allocate_tensor(shape, 2);

    // Fill with non-zero values
    int idx00[] = {0, 0};
    int idx11[] = {1, 1};
    set_tensor_element(t, idx00, 123.45);
    set_tensor_element(t, idx11, 67.89);

    // Ensure they were actually set
    TEST_ASSERT_EQUAL_DOUBLE(123.45, get_tensor_element(t, idx00));

    // Zero the tensor
    tensor_zero(t);

    // Verify all elements (flat check)
    int total_elements = 1;
    for(int i = 0; i < t->ndim; i++) total_elements *= t->shape[i];

    for(int i = 0; i < total_elements; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, t->data[i]);
    }

    free_tensor(t);
}

void test_tensor_zero_integrity(void) {
    int shape[] = {3, 5};
    int ndim = 2;
    tensor_t* t = allocate_tensor(shape, ndim);

    // Capture metadata before zeroing
    int original_stride0 = t->strides[0];
    int original_shape0 = t->shape[0];

    tensor_zero(t);

    // Verify metadata was not corrupted/overwritten
    TEST_ASSERT_EQUAL_INT(ndim, t->ndim);
    TEST_ASSERT_EQUAL_INT(original_shape0, t->shape[0]);
    TEST_ASSERT_EQUAL_INT(original_stride0, t->strides[0]);

    free_tensor(t);
}

// --- Main Runner ---

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_tensor_allocation_success);
    RUN_TEST(test_tensor_get_set_element);
    RUN_TEST(test_tensor_bounds_checking);
    RUN_TEST(test_tensor_dot_product);
    RUN_TEST(test_tensor_dot_product_mismatch);
    RUN_TEST(test_tensor_extract_patch);
    RUN_TEST(test_tensor_extract_patch_invalid);
    RUN_TEST(test_tensor_zero_basic);
    RUN_TEST(test_tensor_zero_integrity);

    return UNITY_END();
}