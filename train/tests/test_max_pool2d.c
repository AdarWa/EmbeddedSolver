//
// Created by adarw on 4/16/26.
//

#include "unity.h"
#include "../src/tensor/tensor.h"
#include "../src/filters/max_pool_2d.h"
#include <math.h>

void setUp(void) {
}

void tearDown(void) {
}

void fill_tensor_channel_2d(tensor_t* t, int c, int height, int width, const double* values) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            set_tensor_element(t, (int[]){i, j, c}, values[i * width + j]);
        }
    }
}

// Test 1: Standard 4x4 input pooled to 2x2 (Single Channel)
void test_max_pool_2d_basic_4x4_to_2x2(void) {
    tensor_t* input = allocate_tensor((int[]){4, 4, 1}, 3);

    const double vals[16] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    fill_tensor_channel_2d(input, 0, 4, 4, vals);

    tensor_t* output = max_pool_2d(input, 2, 2);

    // Assert shape
    TEST_ASSERT_NOT_NULL(output);
    TEST_ASSERT_EQUAL_INT(3, output->ndim);
    TEST_ASSERT_EQUAL_INT(2, output->shape[0]); // Height
    TEST_ASSERT_EQUAL_INT(2, output->shape[1]); // Width
    TEST_ASSERT_EQUAL_INT(1, output->shape[2]); // Channels

    // Assert values
    TEST_ASSERT_EQUAL_DOUBLE(6.0,  get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(8.0,  get_tensor_element(output, (int[]){0, 1, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(14.0, get_tensor_element(output, (int[]){1, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(16.0, get_tensor_element(output, (int[]){1, 1, 0}));

    free_tensor(input);
    free_tensor(output);
}

// Test 2: Multiple Channels (2x2x3 input pooled to 1x1x3)
void test_max_pool_2d_multiple_channels(void) {
    tensor_t* input = allocate_tensor((int[]){2, 2, 3}, 3);

    // Channel 0
    fill_tensor_channel_2d(input, 0, 2, 2, (double[]){1, 2, 3, 4});
    // Channel 1
    fill_tensor_channel_2d(input, 1, 2, 2, (double[]){10, 20, 5, 2});
    // Channel 2
    fill_tensor_channel_2d(input, 2, 2, 2, (double[]){-1, -5, -3, 0});

    tensor_t* output = max_pool_2d(input, 2, 2);

    // Assert shape
    TEST_ASSERT_EQUAL_INT(1, output->shape[0]);
    TEST_ASSERT_EQUAL_INT(1, output->shape[1]);
    TEST_ASSERT_EQUAL_INT(3, output->shape[2]);

    // Assert max values are isolated to their respective channels
    TEST_ASSERT_EQUAL_DOUBLE(4.0,  get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(20.0, get_tensor_element(output, (int[]){0, 0, 1}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0,  get_tensor_element(output, (int[]){0, 0, 2}));

    free_tensor(input);
    free_tensor(output);
}

// Test 3: Handling of negative numbers
// (Important because you initialize max = NAN. Depending on your MAX macro,
// this tests if NAN comparisons break on negative numbers).
void test_max_pool_2d_negative_values(void) {
    tensor_t* input = allocate_tensor((int[]){2, 2, 1}, 3);

    const double vals[4] = {
        -10.0, -15.0,
        -20.0,  -5.0
    };
    fill_tensor_channel_2d(input, 0, 2, 2, vals);

    tensor_t* output = max_pool_2d(input, 2, 2);

    TEST_ASSERT_EQUAL_DOUBLE(-5.0, get_tensor_element(output, (int[]){0, 0, 0}));

    free_tensor(input);
    free_tensor(output);
}

// Test 4: Truncation behavior on odd dimensions (5x5 pooled to 2x2)
void test_max_pool_2d_odd_dimensions_truncation(void) {
    // A 5x5 matrix with a 2x2 pool stride 2 should cleanly drop the 5th row and 5th column
    // because out_height = (5 - 2)/2 + 1 = 2.
    tensor_t* input = allocate_tensor((int[]){5, 5, 1}, 3);

    const double vals[25] = {
        1, 1, 2, 2, 99,
        1, 1, 2, 2, 99,
        3, 3, 4, 4, 99,
        3, 3, 4, 4, 99,
       99,99,99,99, 99
    };
    fill_tensor_channel_2d(input, 0, 5, 5, vals);

    tensor_t* output = max_pool_2d(input, 2, 2);

    // Check that it ignores the 99s in the truncated 5th row/col
    TEST_ASSERT_EQUAL_INT(2, output->shape[0]);
    TEST_ASSERT_EQUAL_INT(2, output->shape[1]);

    TEST_ASSERT_EQUAL_DOUBLE(1.0, get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(2.0, get_tensor_element(output, (int[]){0, 1, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(3.0, get_tensor_element(output, (int[]){1, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(4.0, get_tensor_element(output, (int[]){1, 1, 0}));

    free_tensor(input);
    free_tensor(output);
}

// Test 5: 1x1 Pooling (Identity Operation)
void test_max_pool_2d_1x1_pool(void) {
    tensor_t* input = allocate_tensor((int[]){3, 3, 1}, 3);

    const double vals[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    fill_tensor_channel_2d(input, 0, 3, 3, vals);

    tensor_t* output = max_pool_2d(input, 1, 1);

    TEST_ASSERT_EQUAL_INT(3, output->shape[0]);
    TEST_ASSERT_EQUAL_INT(3, output->shape[1]);

    // Ensure values are identical to input
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            double expected = vals[i * 3 + j];
            double actual = get_tensor_element(output, (int[]){i, j, 0});
            TEST_ASSERT_EQUAL_DOUBLE(expected, actual);
        }
    }

    free_tensor(input);
    free_tensor(output);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_max_pool_2d_basic_4x4_to_2x2);
    RUN_TEST(test_max_pool_2d_multiple_channels);
    RUN_TEST(test_max_pool_2d_negative_values);
    RUN_TEST(test_max_pool_2d_odd_dimensions_truncation);
    RUN_TEST(test_max_pool_2d_1x1_pool);

    return UNITY_END();
}