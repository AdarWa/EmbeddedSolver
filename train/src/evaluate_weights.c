//
// Created by adarw on 4/20/26.
//

#include "dataset/img_handler/img_handler.h"
#include "dataset/remote/remote.h"
#include "evaluation/evaluation.h"
#include "logging/logging.h"
#include "model/allocations.h"

static const char* TAG = "evaluate_weights";

static const model_train_config_t model_train_config = {
    .epochs = 10,
    .batch_size = 120,
    .max_pool_size = 2,
    .channel_cnt = 1,
    .img_size = 28,
    .filter_cnt_1 = 32,
    .filter_cnt_2 = 64,
    .kernel_size = 3,
    .dense_cnt_1 = 128,
    .dense_cnt_2 = 64,
    .dense_cnt_3 = 10,
    .dropout_rate_1 = 0.2,
    .dropout_rate_2 = 0.2
};

int main() {
    set_log_level(L_INFO);
    if (!remote_all_data_exists()) {
        remote_init();
        remote_fetch_all();
        remote_cleanup();
    }

    mnist_dataset_t test_data = mnist_parse_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    LOG_I(TAG, "Loaded Dataset! Test: %d",test_data.count);

    weights_t weights = {0};
    load_weights(&weights);

    LOG_I(TAG, "Accuracy: %d", evaluate_accuracy(model_train_config, &weights, &test_data));

    return 0;
}
