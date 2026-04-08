//
// Created by adarw on 4/8/26.
//

#include "draw.h"

#include <esp_lcd_touch.h>

#include "graphic.h"

void handle_draw(esp_lcd_touch_point_data_t data) {
    put_draw_buff(data.x, data.y, BRUSH_COLOR);
}
