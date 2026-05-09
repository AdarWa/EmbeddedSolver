//
// Created by adarw on 4/16/26.
//

#include "unity.h"
#include "../src/tensor/tensor.h"
#include "../src/filters/max_pool_2d.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

// Helper to fill a tensor channel
void fill_tensor_channel_2d(tensor_t* t, int c, int height, int width, const double* values) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            set_tensor_element(t, (int[]){i, j, c}, values[i * width + j]);
        }
    }
}

// Test 1: Forward Pass & Coordinate Mask Accuracy
void test_max_pool_2d_forward_and_indices(void) {
    tensor_t* input = allocate_tensor((int[]){4, 4, 1}, 3);
    const double vals[16] = {
         1,  5,  3,  4,
         2,  4,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    fill_tensor_channel_2d(input, 0, 4, 4, vals);

    tensor_t *h_idx = NULL, *w_idx = NULL;
    // Calling updated function that returns masks
    tensor_t* output = max_pool_2d_forward(input, 2, 2, &h_idx, &w_idx);

    TEST_ASSERT_NOT_NULL(output);
    TEST_ASSERT_NOT_NULL(h_idx);
    TEST_ASSERT_NOT_NULL(w_idx);

    // Assert forward values for the first 2x2 block (max is 5.0 at index 0,1)
    TEST_ASSERT_EQUAL_DOUBLE(5.0, get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(h_idx, (int[]){0, 0, 0})); // Row 0
    TEST_ASSERT_EQUAL_DOUBLE(1.0, get_tensor_element(w_idx, (int[]){0, 0, 0})); // Col 1

    // Assert forward values for the last 2x2 block (max is 16.0 at index 3,3)
    TEST_ASSERT_EQUAL_DOUBLE(16.0, get_tensor_element(output, (int[]){1, 1, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(3.0,  get_tensor_element(h_idx, (int[]){1, 1, 0})); // Row 3
    TEST_ASSERT_EQUAL_DOUBLE(3.0,  get_tensor_element(w_idx, (int[]){1, 1, 0})); // Col 3

    free_tensor(input);
    free_tensor(output);
    free_tensor(h_idx);
    free_tensor(w_idx);
}

// Test 2: Multiple Channels
void test_max_pool_2d_multiple_channels(void) {
    tensor_t* input = allocate_tensor((int[]){2, 2, 2}, 3);
    fill_tensor_channel_2d(input, 0, 2, 2, (double[]){1, 2, 3, 4});    // Max 4 at (1,1)
    fill_tensor_channel_2d(input, 1, 2, 2, (double[]){10, 5, 2, 1});  // Max 10 at (0,0)

    tensor_t *h_idx = NULL, *w_idx = NULL;
    tensor_t* output = max_pool_2d_forward(input, 2, 2, &h_idx, &w_idx);

    TEST_ASSERT_EQUAL_DOUBLE(4.0,  get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(1.0,  get_tensor_element(h_idx, (int[]){0, 0, 0}));

    TEST_ASSERT_EQUAL_DOUBLE(10.0, get_tensor_element(output, (int[]){0, 0, 1}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0,  get_tensor_element(h_idx, (int[]){0, 0, 1}));

    free_tensor(input);
    free_tensor(output);
    free_tensor(h_idx);
    free_tensor(w_idx);
}

// Test 3: Backward Pass Logic (Gradient Routing)
void test_max_pool_2d_backward_routing(void) {
    int in_h = 4, in_w = 4, in_c = 1;
    tensor_t* input = allocate_tensor((int[]){in_h, in_w, in_c}, 3);
    // Max values are intentionally placed at (0,1), (0,3), (3,0), (3,3)
    const double vals[16] = {
        0, 10,  0, 10,
        0,  0,  0,  0,
        0,  0,  0,  0,
       10,  0,  0, 10
    };
    fill_tensor_channel_2d(input, 0, 4, 4, vals);

    // 1. Forward Pass to get masks
    tensor_t *h_idx = NULL, *w_idx = NULL;
    tensor_t* output = max_pool_2d_forward(input, 2, 2, &h_idx, &w_idx);

    // 2. Create dummy upstream gradient (d_loss/d_output)
    // We'll say the error is 5.0 for every pooled pixel
    tensor_t* d_upstream = allocate_tensor((int[]){2, 2, 1}, 3);
    for(int i=0; i<4; i++) d_upstream->data[i] = 5.0;

    // 3. Run Backward Pass
    tensor_t* d_input = max_pool_2d_backward(d_upstream, h_idx, w_idx, in_h, in_w, in_c);

    // 4. Verify Gradient Routing
    // The gradient should ONLY exist at the winning coordinates
    TEST_ASSERT_EQUAL_DOUBLE(5.0, get_tensor_element(d_input, (int[]){0, 1, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(5.0, get_tensor_element(d_input, (int[]){0, 3, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(5.0, get_tensor_element(d_input, (int[]){3, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(5.0, get_tensor_element(d_input, (int[]){3, 3, 0}));

    // Non-winning pixels must have 0.0 gradient
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(d_input, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(d_input, (int[]){1, 1, 0}));

    free_tensor(input);
    free_tensor(output);
    free_tensor(h_idx);
    free_tensor(w_idx);
    free_tensor(d_upstream);
    free_tensor(d_input);
}

// Test 4: Handling Negative Values (Initialization Check)
void test_max_pool_2d_negative_values(void) {
    tensor_t* input = allocate_tensor((int[]){2, 2, 1}, 3);
    fill_tensor_channel_2d(input, 0, 2, 2, (double[]){-10, -5, -20, -15}); // Max -5 at (0,1)

    tensor_t *h_idx = NULL, *w_idx = NULL;
    tensor_t* output = max_pool_2d_forward(input, 2, 2, &h_idx, &w_idx);

    TEST_ASSERT_EQUAL_DOUBLE(-5.0, get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0,  get_tensor_element(h_idx, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(1.0,  get_tensor_element(w_idx, (int[]){0, 0, 0}));

    free_tensor(input);
    free_tensor(output);
    free_tensor(h_idx);
    free_tensor(w_idx);
}

// Test 5: Odd Dimension Truncation
void test_max_pool_2d_truncation(void) {
    tensor_t* input = allocate_tensor((int[]){3, 3, 1}, 3);
    const double vals[9] = {
        10, 2, 99,
         3, 4, 99,
        99,99, 99
    };
    fill_tensor_channel_2d(input, 0, 3, 3, vals);

    tensor_t *h_idx = NULL, *w_idx = NULL;
    tensor_t* output = max_pool_2d_forward(input, 2, 2, &h_idx, &w_idx);

    // Should be a 1x1 output (3x3 input, 2x2 pool, stride 2)
    TEST_ASSERT_EQUAL_INT(1, output->shape[0]);
    TEST_ASSERT_EQUAL_INT(1, output->shape[1]);
    TEST_ASSERT_EQUAL_DOUBLE(10.0, get_tensor_element(output, (int[]){0, 0, 0}));

    free_tensor(input);
    free_tensor(output);
    free_tensor(h_idx);
    free_tensor(w_idx);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_max_pool_2d_forward_and_indices);
    RUN_TEST(test_max_pool_2d_multiple_channels);
    RUN_TEST(test_max_pool_2d_backward_routing);
    RUN_TEST(test_max_pool_2d_negative_values);
    RUN_TEST(test_max_pool_2d_truncation);
    return UNITY_END();
}