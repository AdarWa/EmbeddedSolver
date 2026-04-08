//
// Created by adarw on 4/8/26.
//

#include "draw.h"

#include <esp_lcd_touch.h>

#include "graphic.h"

void handle_draw(esp_lcd_touch_point_data_t data) {
    int x1 = CLIP(data.x - BRUSH_RADIUS/2, 0, LCD_H_RES);
    int y1 = CLIP(data.y - BRUSH_RADIUS/2, 0, LCD_V_RES);
    int x2 = CLIP(data.x + BRUSH_RADIUS/2, 0, LCD_H_RES);
    int y2 = CLIP(data.y + BRUSH_RADIUS/2, 0, LCD_V_RES);
    fill_draw_buff(x1, y1, x2, y2, BRUSH_COLOR);
}
