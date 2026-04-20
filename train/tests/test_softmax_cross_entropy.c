//
// Created by adarw on 4/20/26.
//

#include "unity.h"
#include "../src/tensor/tensor.h"
#include "../src/model/backpropagation.h"

void setUp(void) {}
void tearDown(void) {}

// Test 1: Perfect Prediction
// If the network is 100% sure and correct, gradients should be 0.
void test_loss_gradient_perfect_prediction(void) {
    tensor_t* predictions = allocate_tensor((int[]){3}, 1);
    set_tensor_element(predictions, (int[]){0}, 0.0);
    set_tensor_element(predictions, (int[]){1}, 1.0); // Correct class
    set_tensor_element(predictions, (int[]){2}, 0.0);

    tensor_t* d_logits = compute_softmax_cross_entropy_gradient(predictions, 1);

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, get_tensor_element(d_logits, (int[]){i}));
    }

    free_tensor(predictions);
    free_tensor(d_logits);
}

// Test 2: Catastrophic Confidence (Wrong Answer)
// Network is 100% sure it's index 0, but it's actually index 2.
void test_loss_gradient_confidently_wrong(void) {
    tensor_t* predictions = allocate_tensor((int[]){3}, 1);
    set_tensor_element(predictions, (int[]){0}, 1.0); // Wrongly confident
    set_tensor_element(predictions, (int[]){1}, 0.0);
    set_tensor_element(predictions, (int[]){2}, 0.0); // Target is here

    tensor_t* d_logits = compute_softmax_cross_entropy_gradient(predictions, 2);

    // Index 0: (1.0 - 0.0) = 1.0 (Strong "Decrease this score" signal)
    TEST_ASSERT_EQUAL_DOUBLE(1.0, get_tensor_element(d_logits, (int[]){0}));
    // Index 2: (0.0 - 1.0) = -1.0 (Strong "Increase this score" signal)
    TEST_ASSERT_EQUAL_DOUBLE(-1.0, get_tensor_element(d_logits, (int[]){2}));

    free_tensor(predictions);
    free_tensor(d_logits);
}

// Test 3: Total Uncertainty (Uniform Distribution)
// Predictions are [0.33, 0.33, 0.33]
void test_loss_gradient_uniform_uncertainty(void) {
    tensor_t* predictions = allocate_tensor((int[]){3}, 1);
    for(int i=0; i<3; i++) set_tensor_element(predictions, (int[]){i}, 1.0/3.0);

    tensor_t* d_logits = compute_softmax_cross_entropy_gradient(predictions, 0);

    // Target index 0: 0.333 - 1.0 = -0.666...
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.666666, get_tensor_element(d_logits, (int[]){0}));
    // Other indices: 0.333 - 0.0 = 0.333...
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.333333, get_tensor_element(d_logits, (int[]){1}));

    free_tensor(predictions);
    free_tensor(d_logits);
}

// Test 4: MNIST Scale (10 Classes)
void test_loss_gradient_mnist_scale(void) {
    int classes = 10;
    tensor_t* predictions = allocate_tensor((int[]){classes}, 1);
    // Simulate some random softmax output that sums to 1.0
    for(int i=0; i<classes; i++) set_tensor_element(predictions, (int[]){i}, 0.1);

    int target = 7; // The digit 7
    tensor_t* d_logits = compute_softmax_cross_entropy_gradient(predictions, target);

    TEST_ASSERT_EQUAL_INT(10, d_logits->shape[0]);
    // Check index 7
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.9, get_tensor_element(d_logits, (int[]){target}));
    // Check index 0
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.1, get_tensor_element(d_logits, (int[]){0}));

    free_tensor(predictions);
    free_tensor(d_logits);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_loss_gradient_perfect_prediction);
    RUN_TEST(test_loss_gradient_confidently_wrong);
    RUN_TEST(test_loss_gradient_uniform_uncertainty);
    RUN_TEST(test_loss_gradient_mnist_scale);
    return UNITY_END();
}