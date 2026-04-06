//
// Created by adarw on 4/1/26.
//

#pragma once

#include <esp_chip_info.h>

static inline void print_info() {
    const char* TAG = "chip_info";
    esp_chip_info_t chip_info;
    uint32_t flash_size;
    esp_chip_info(&chip_info);

    if(esp_flash_get_size(nullptr, &flash_size) != ESP_OK) {
        flash_size = 0;
    }

    ESP_LOGI(TAG, "------------------------------------------");
    ESP_LOGI(TAG, "Model: %s", chip_info.model == CHIP_ESP32 ? "ESP32" : "Unknown");
    ESP_LOGI(TAG, "Cores: %d", chip_info.cores);
    ESP_LOGI(TAG, "Revision: %d", chip_info.revision);

    ESP_LOGI(TAG, "Features:%s%s%s",
             (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? " WiFi" : "",
             (chip_info.features & CHIP_FEATURE_BT) ? " BT" : "",
             (chip_info.features & CHIP_FEATURE_BLE) ? " BLE" : "");

    ESP_LOGI(TAG, "Flash: %lu MB %s", flash_size / (1024 * 1024),
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");
    ESP_LOGI(TAG, "------------------------------------------");
}
