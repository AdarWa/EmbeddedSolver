//
// Created by adarw on 4/6/26.
//

#include "graphic.h"

#include "esp_check.h"
#include "esp_heap_caps.h"
#include "lcd/lcd.h"

static const char* TAG = "graphic";

static uint16_t* drawing_buffer = NULL;

esp_err_t allocate_drawing_buff() {
    drawing_buffer = heap_caps_malloc(DRAW_BUFF_SIZE * sizeof(uint16_t), MALLOC_CAP_DMA);
    ESP_RETURN_ON_FALSE(drawing_buffer, ESP_ERR_NO_MEM, TAG, "no memory in order to allocate drawing buffer!");
    return ESP_OK;
}

void put_draw_buff(uint16_t x, uint16_t y, uint16_t color) {
    if (drawing_buffer) {
        drawing_buffer[CALC_DRAW_INDEX(x,y)] = color;
    }
}

void fill_draw_buff(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t color) {
    for (uint16_t y = y1; y <= y2; y++) {
        for (uint16_t x = x1; x <= x2; x++) {
            put_draw_buff(x,y, color);
        }
    }
}

void clear_draw_buff() {
    for (int i = 0; i < DRAW_BUFF_SIZE; i++) {
        drawing_buffer[i] = 0x00;
    }
}

esp_err_t draw_buff(esp_lcd_panel_handle_t handle) {
    ESP_RETURN_ON_FALSE(drawing_buffer, ESP_ERR_NO_MEM, TAG, "no memory to draw buffer!");
    lcd_draw(handle, 0, 0, DRAW_BUFF_X, DRAW_BUFF_Y, drawing_buffer);
    return ESP_OK;
}