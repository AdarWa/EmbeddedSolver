//
// Created by adarw on 4/16/26.
//

#include "unity.h"
#include <stdlib.h>

#include "../src/tensor/tensor.h"
#include "../src/filters/conv_2d.h"

void populate_tensor_3d(tensor_t* t, double* values) {
    int h = t->shape[0];
    int w = t->shape[1];
    int c = t->shape[2];
    int idx = 0;
    for(int y=0; y<h; y++) {
        for(int x=0; x<w; x++) {
            for(int ch=0; ch<c; ch++) {
                set_tensor_element(t, (int[]){y, x, ch}, values[idx++]);
            }
        }
    }
}

void setUp(void) {
}

void tearDown(void) {
}


void test_conv2d_forward_pass_basic_2x2_kernel(void) {
    // 1. Setup Input: 3x3x1 image
    tensor_t* input = allocate_tensor((int[]){3, 3, 1}, 3);
    double in_data[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    populate_tensor_3d(input, in_data);

    // 2. Setup Weights: 1 filter of 2x2x1
    int num_filters = 1;
    int kernel_size = 2;
    tensor_t* filter0 = allocate_tensor((int[]){2, 2, 1}, 3);
    double f0_data[] = {
        1, 0,
        0, 1
    };
    populate_tensor_3d(filter0, f0_data);
    tensor_t* weights[] = { filter0 };

    // 3. Setup Biases
    double biases[] = { 1.0 };

    // 4. Execute
    tensor_t* output = conv2d_forward_pass(input, weights, biases, num_filters, kernel_size);

    // 5. Assert Output Shape (Expected: 2x2x1)
    TEST_ASSERT_NOT_NULL(output);
    TEST_ASSERT_EQUAL_INT(3, output->ndim);
    TEST_ASSERT_EQUAL_INT(2, output->shape[0]);
    TEST_ASSERT_EQUAL_INT(2, output->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, output->shape[2]);

    // 6. Assert Mathematics
    // Expected output mapping (Input * Filter + Bias) -> ReLU:
    // y0, x0: (1*1 + 5*1) + 1.0 = 7.0
    // y0, x1: (2*1 + 6*1) + 1.0 = 9.0
    // y1, x0: (4*1 + 8*1) + 1.0 = 13.0
    // y1, x1: (5*1 + 9*1) + 1.0 = 15.0
    TEST_ASSERT_EQUAL_DOUBLE(7.0,  get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(9.0,  get_tensor_element(output, (int[]){0, 1, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(13.0, get_tensor_element(output, (int[]){1, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(15.0, get_tensor_element(output, (int[]){1, 1, 0}));

    // 7. Cleanup
    free_tensor(input);
    free_tensor(filter0);
    free_tensor(output);
}

void test_conv2d_forward_pass_relu_activation(void) {
    // 1. Setup Input: 2x2x1 image with negative values
    tensor_t* input = allocate_tensor((int[]){2, 2, 1}, 3);
    double in_data[] = {
        -5, -10,
         2, -3
    };
    populate_tensor_3d(input, in_data);

    // 2. Setup Weights: 1 filter of 1x1x1 (Identity)
    int num_filters = 1;
    int kernel_size = 1;
    tensor_t* filter0 = allocate_tensor((int[]){1, 1, 1}, 3);
    set_tensor_element(filter0, (int[]){0, 0, 0}, 1.0);
    tensor_t* weights[] = { filter0 };

    // 3. Setup Biases
    double biases[] = { 0.0 };

    // 4. Execute
    tensor_t* output = conv2d_forward_pass(input, weights, biases, num_filters, kernel_size);

    // 5. Assert Logic (ReLU should clip negative values to 0)
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(output, (int[]){0, 0, 0})); // -5 -> 0
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(output, (int[]){0, 1, 0})); // -10 -> 0
    TEST_ASSERT_EQUAL_DOUBLE(2.0, get_tensor_element(output, (int[]){1, 0, 0})); // 2 -> 2
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(output, (int[]){1, 1, 0})); // -3 -> 0

    // 6. Cleanup
    free_tensor(input);
    free_tensor(filter0);
    free_tensor(output);
}

void test_conv2d_forward_pass_multiple_filters(void) {
    // 1. Setup Input: 2x2x2 (2 channels)
    tensor_t* input = allocate_tensor((int[]){2, 2, 2}, 3);
    double in_data[] = {
        1, 1,  2, 2,
        3, 3,  4, 4
    };
    populate_tensor_3d(input, in_data);

    // 2. Setup Weights: 2 filters of 1x1x2
    int num_filters = 2;
    int kernel_size = 1;

    // Filter 0 sums channels directly
    tensor_t* filter0 = allocate_tensor((int[]){1, 1, 2}, 3);
    set_tensor_element(filter0, (int[]){0,0,0}, 1.0);
    set_tensor_element(filter0, (int[]){0,0,1}, 1.0);

    // Filter 1 subtracts channel 1 from channel 0
    tensor_t* filter1 = allocate_tensor((int[]){1, 1, 2}, 3);
    set_tensor_element(filter1, (int[]){0,0,0}, 1.0);
    set_tensor_element(filter1, (int[]){0,0,1}, -1.0);

    tensor_t* weights[] = { filter0, filter1 };
    double biases[] = { 0.0, 0.0 };

    // 3. Execute
    tensor_t* output = conv2d_forward_pass(input, weights, biases, num_filters, kernel_size);

    // 4. Assert Output Shape (Expected: 2x2x2)
    TEST_ASSERT_EQUAL_INT(2, output->shape[0]);
    TEST_ASSERT_EQUAL_INT(2, output->shape[1]);
    TEST_ASSERT_EQUAL_INT(2, output->shape[2]); // Num filters

    // 5. Assert Output Math
    // Filter 0 (Index 0): 1+1=2, 2+2=4, 3+3=6, 4+4=8
    TEST_ASSERT_EQUAL_DOUBLE(2.0, get_tensor_element(output, (int[]){0, 0, 0}));
    TEST_ASSERT_EQUAL_DOUBLE(8.0, get_tensor_element(output, (int[]){1, 1, 0}));

    // Filter 1 (Index 1): 1-1=0, 2-2=0, 3-3=0, 4-4=0
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(output, (int[]){0, 0, 1}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(output, (int[]){1, 1, 1}));

    free_tensor(input);
    free_tensor(filter0);
    free_tensor(filter1);
    free_tensor(output);
}

void test_conv2d_backward_pass_basic(void) {
    // 1. Setup Input: 3x3x1 image of all 1.0s
    tensor_t* input = allocate_tensor((int[]){3, 3, 1}, 3);
    for(int i=0; i<9; i++) input->data[i] = 1.0;

    // 2. Setup Weights: One 2x2x1 filter of all 2.0s
    int num_filters = 1;
    int kernel_size = 2;
    tensor_t* filter0 = allocate_tensor((int[]){2, 2, 1}, 3);
    for(int i=0; i<4; i++) filter0->data[i] = 2.0;
    tensor_t* weights[] = { filter0 };

    // 3. Setup Biases: 0.0
    double biases[] = { 0.0 };

    // 4. Run Forward Pass (to get output_forward needed for ReLU check)
    tensor_t* output_forward = conv2d_forward_pass(input, weights, biases, num_filters, kernel_size);

    // 5. Setup Upstream Gradient (d_out): 2x2x1 of all 5.0s
    tensor_t* d_out = allocate_tensor((int[]){2, 2, 1}, 3);
    for(int i=0; i<4; i++) d_out->data[i] = 5.0;

    // 6. Setup Gradient Accumulators (Initialized to 0)
    tensor_t* d_filter0 = allocate_tensor((int[]){2, 2, 1}, 3);
    tensor_t* d_weights[] = { d_filter0 };
    double d_biases[] = { 0.0 };

    // 7. Execute Backward Pass
    tensor_t* d_input = conv2d_backward_pass(d_out, input, output_forward, weights,
                                             d_weights, d_biases, num_filters, kernel_size);

    // --- ASSERTIONS ---

    // Bias Gradient: Sum of upstream gradients (5.0 * 4 pixels = 20.0)
    TEST_ASSERT_EQUAL_DOUBLE(20.0, d_biases[0]);

    // Weight Gradient: pixel_val (1.0) * upstream_grad (5.0) * 4 window positions = 20.0
    TEST_ASSERT_EQUAL_DOUBLE(20.0, get_tensor_element(d_weights[0], (int[]){0, 0, 0}));

    // Input Gradient: The center pixel (1,1) is covered by all four 2x2 window positions.
    // Each position contributes (weight=2.0 * upstream_grad=5.0) = 10.0.
    // Total for center pixel should be 10.0 * 4 = 40.0.
    TEST_ASSERT_EQUAL_DOUBLE(40.0, get_tensor_element(d_input, (int[]){1, 1, 0}));

    // Edge pixel (0,0) is only covered by one window position.
    // Total should be 2.0 * 5.0 = 10.0.
    TEST_ASSERT_EQUAL_DOUBLE(10.0, get_tensor_element(d_input, (int[]){0, 0, 0}));

    // Cleanup
    free_tensor(input);
    free_tensor(filter0);
    free_tensor(output_forward);
    free_tensor(d_out);
    free_tensor(d_filter0);
    free_tensor(d_input);
}

void test_conv2d_backward_relu_blocking(void) {
    // Input is positive, but weight is negative enough to make Z negative.
    tensor_t* input = allocate_tensor((int[]){1, 1, 1}, 3);
    set_tensor_element(input, (int[]){0,0,0}, 10.0);

    tensor_t* filter0 = allocate_tensor((int[]){1, 1, 1}, 3);
    set_tensor_element(filter0, (int[]){0,0,0}, -1.0); // 10 * -1 = -10

    tensor_t* weights[] = { filter0 };
    double biases[] = { 0.0 };

    // Forward pass: Result is MAX(0, -10) = 0.0
    tensor_t* output_forward = conv2d_forward_pass(input, weights, biases, 1, 1);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(output_forward, (int[]){0,0,0}));

    // Upstream Gradient is 100.0
    tensor_t* d_out = allocate_tensor((int[]){1, 1, 1}, 3);
    set_tensor_element(d_out, (int[]){0,0,0}, 100.0);

    tensor_t* d_filter0 = allocate_tensor((int[]){1, 1, 1}, 3);
    double d_biases[] = { 0.0 };

    // Execute Backward
    tensor_t* d_input = conv2d_backward_pass(d_out, input, output_forward, weights,
                                             (tensor_t*[]){d_filter0}, d_biases, 1, 1);

    // Because output_forward was 0, ALL gradients for this pixel should be 0.
    TEST_ASSERT_EQUAL_DOUBLE(0.0, d_biases[0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(d_filter0, (int[]){0,0,0}));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(d_input, (int[]){0,0,0}));

    free_tensor(input);
    free_tensor(filter0);
    free_tensor(output_forward);
    free_tensor(d_out);
    free_tensor(d_filter0);
    free_tensor(d_input);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_conv2d_forward_pass_basic_2x2_kernel);
    RUN_TEST(test_conv2d_forward_pass_relu_activation);
    RUN_TEST(test_conv2d_forward_pass_multiple_filters);
    RUN_TEST(test_conv2d_backward_pass_basic);
    RUN_TEST(test_conv2d_backward_relu_blocking);
    return UNITY_END();
}