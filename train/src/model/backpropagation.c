//
// Created by adarw on 4/20/26.
//

#include "backpropagation.h"

#include <assert.h>

#include "tensor/tensor.h"

tensor_t* compute_softmax_cross_entropy_gradient(const tensor_t* predictions, int target_label) {
    assert(predictions->ndim == 1);
    int num_classes = predictions->shape[0];

    tensor_t* d_logits = allocate_tensor((int[]){num_classes}, 1);

    for (int i = 0; i < num_classes; i++) {
        double y_i = get_tensor_element(predictions, (int[]){i});

        double t_i = (i == target_label) ? 1.0 : 0.0;

        double gradient = y_i - t_i;

        set_tensor_element(d_logits, (int[]){i}, gradient);
    }

    return d_logits;
}
