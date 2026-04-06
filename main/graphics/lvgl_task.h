//
// Created by adarw on 4/1/26.
//

#pragma once
#include "esp_err.h"

#define LVGL_TASK_NAME "lvgl"
#define NEW_SCR_ANIMATION_DELAY 250
#define LVGL_STACK_SIZE 4096 // Try increasing this if having stackoverflow errors

esp_err_t configureLvgl();
esp_err_t startLvglTask();
esp_err_t lock_lvgl_mutex();
void unlock_lvgl_mutex();