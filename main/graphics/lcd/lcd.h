//
// Created by adarw on 4/1/26.
//

#pragma once
#include "esp_err.h"
#include "esp_lcd_types.h"

esp_err_t lcd_display_brightness_init(void);
esp_err_t lcd_display_brightness_set(int );
esp_err_t lcd_display_backlight_off(void);
esp_err_t lcd_display_backlight_on(void);

esp_err_t lcd_init(esp_lcd_panel_io_handle_t *, esp_lcd_panel_handle_t *);
esp_err_t lcd_draw(esp_lcd_panel_handle_t panel, int x1, int y1, int x2, int y2,
                               const uint16_t* color_data);