//
// Created by adarw on 4/1/26.
//

#pragma once
#include <esp_lcd_touch.h>
#include <esp_lcd_types.h>

#include "esp_err.h"

#define GRAPHIC_TASK_NAME "task/graphic"
#define GRAPHIC_STACK_SIZE 4096 // Try increasing this if having stackoverflow errors
#define GRAPHIC_TASK_LOOP_TIME 3 // ms

typedef struct {
    esp_lcd_panel_handle_t lcd_handle;
    esp_lcd_touch_handle_t touch_handle;
} graphic_task_params_t;

esp_err_t configure_lcd(esp_lcd_panel_handle_t* handle);
esp_err_t configure_touch(esp_lcd_touch_handle_t* handle);
esp_err_t start_graphic_task();
esp_err_t lock_graphic_mutex();
void unlock_graphic_mutex();