//
// Created by adarw on 4/9/26.
//

#include "dataset/img_handler/img_handler.h"
#include "dataset/remote/remote.h"
#include "logging/logging.h"

static const char* TAG = "main";

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



    // Cleanup
    mnist_free_dataset(&train_data);
    mnist_free_dataset(&test_data);

    return 0;
}