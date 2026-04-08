//
// Created by adarw on 4/8/26.
//

#pragma once

#include <esp_lcd_touch.h>

#define BRUSH_COLOR 0x07E0
#define BRUSH_RADIUS 10
#define CLIP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

void handle_draw(esp_lcd_touch_point_data_t data);