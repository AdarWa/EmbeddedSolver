//
// Created by adarw on 4/8/26.
//

#pragma once

#include <math.h>
#include <esp_lcd_touch.h>

#define BRUSH_COLOR 0x07E0
#define BRUSH_DIAMETER 10
#define CLIP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define DEG_TO_RAD (M_PI / 180.0)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void handle_draw(esp_lcd_touch_point_data_t data, uint8_t point_cnt);