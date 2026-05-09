//
// Created by adarw on 4/9/26.
//

#pragma once

#define INPUT_SHAPE 28 // (28,28,1)
#define OUTPUT_SHAPE 10
#define BATCH_SIZE 128
#define EPOCHS 10

typedef enum {
    SIGMOID,
    RELU,
    TANH
} activation_type;

typedef struct {
    double input;
    double output;
    double bias;
    double gradient;
    activation_type activation;
} neuron_t;

typedef struct {
    int neuron_cnt;
    neuron_t* neurons;
} layer_t;