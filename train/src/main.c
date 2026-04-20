//
// Created by adarw on 4/9/26.
//

#include "dataset/img_handler/img_handler.h"
#include "dataset/remote/remote.h"
#include "logging/logging.h"
#include "model/model.h"

static const char* TAG = "main";
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
    if (!remote_all_data_exists()) {
        remote_init();
        remote_fetch_all();
        remote_cleanup();
    }

    mnist_dataset_t train_data = mnist_parse_dataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    mnist_dataset_t test_data = mnist_parse_dataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

    LOG_I(TAG, "Loaded Dataset! Train: %d, Test: %d", train_data.count, test_data.count);

    if (train_data.count < 1 || test_data.count < 1) {
        mnist_free_dataset(&train_data);
        mnist_free_dataset(&test_data);
        LOG_F(TAG, "Found invalid dataset count!");
    }

    train_model(model_train_config, &train_data);

    // Cleanup
    mnist_free_dataset(&train_data);
    mnist_free_dataset(&test_data);

    return 0;
}