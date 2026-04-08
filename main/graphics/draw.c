//
// Created by adarw on 4/8/26.
//

#include "draw.h"

#include <esp_lcd_touch.h>

#include "graphic.h"

static void draw_point(uint16_t x0, uint16_t y0) {
    int r = BRUSH_DIAMETER / 2;
    int x = r;
    int y = 0;
    int err = 0;

    while (x >= y) {
        for (int i = x0 - x; i <= x0 + x; i++) {
            put_draw_buff(CLIP(i, 0, LCD_H_RES), CLIP(y0 + y, 0, LCD_V_RES), BRUSH_COLOR);
            put_draw_buff(CLIP(i, 0, LCD_H_RES), CLIP(y0 - y, 0, LCD_V_RES), BRUSH_COLOR);
        }
        for (int i = x0 - y; i <= x0 + y; i++) {
            put_draw_buff(CLIP(i, 0, LCD_H_RES), CLIP(y0 + x, 0, LCD_V_RES), BRUSH_COLOR);
            put_draw_buff(CLIP(i, 0, LCD_H_RES), CLIP(y0 - x, 0, LCD_V_RES), BRUSH_COLOR);
        }

        if (err <= 0) {
            y += 1;
            err += 2 * y + 1;
        }
        if (err > 0) {
            x -= 1;
            err -= 2 * x + 1;
        }
    }
}

static void draw_line(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (1) {
        draw_point(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void handle_draw(esp_lcd_touch_point_data_t data) {

    static esp_lcd_touch_point_data_t last_point;
    static bool first_touch = true;

    if (first_touch) {
        last_point = data;
        first_touch = false;
    }
    draw_line(last_point.x, last_point.y, data.x, data.y);
    last_point = data;
}
