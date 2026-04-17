//
// Created by adarw on 4/16/26.
//

#include "model.h"

#include "logging/logging.h"
#include "filters/conv_2d.h"

static const char* TAG = "model";

tensor_t* process_model_step(tensor_t* input_tensor, model_step_t step) {
    switch (step.type) {
    case CONV_2D:
        LOG_I(TAG, "processing model step of type CONV_2D with %d filters, and kernel size (h,w)(%d,%d)",
              step.conv2d.filters, step.conv2d.kernel_height, step.conv2d.kernel_width);

        return conv2d_forward_pass(input_tensor, );

    case MAX_POOL_2D:
        LOG_I(TAG, "processing model step of type MAX_POOL_2D with %d pool size", step.max_pool.pool_size);
        break;

    case FLATTEN:
        LOG_I(TAG, "processing model step of type FLATTEN");
        break;

    case DENSE:
        LOG_I(TAG, "processing model step of type DENSE with %d units", step.dense.units);
        break;

    case DROPOUT:
        LOG_I(TAG, "processing model step of type DROPOUT with %f rate", step.dropout.rate);
        break;

    }
    return NULL;
}
