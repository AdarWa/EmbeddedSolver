//
// Created by adarw on 4/6/26.
//

#pragma once

#include "lcd/hardware.h"

#define DRAW_BUFF_X LCD_V_RES
#define DRAW_BUFF_Y LCD_H_RES
#define DRAW_BUFF_SIZE (DRAW_BUFF_X*DRAW_BUFF_Y)
#define CALC_DRAW_INDEX(x, y) (((x) * DRAW_BUFF_X) + (y))

#include <esp_lcd_types.h>

#include "esp_err.h"

esp_err_t allocate_drawing_buff();
void fill_draw_buff(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t color);
void put_draw_buff(uint16_t x, uint16_t y, uint16_t color);
void clear_draw_buff();
esp_err_t draw_buff(esp_lcd_panel_handle_t handle);
