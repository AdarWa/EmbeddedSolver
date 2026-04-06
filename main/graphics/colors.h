//
// Created by adarw on 4/1/26.
//

#pragma once

#include <lvgl.h>

#define LV_CLR_WHITE            lv_color_hex(0xFFFFFF)
#define LV_CLR_BLACK            lv_color_hex(0x000000)
#define LV_CLR_TRANSPARENT      lv_color_hex(0x000000)

#define LV_CLR_RED              lv_color_hex(0xFF0000)
#define LV_CLR_MAROON           lv_color_hex(0x800000)
#define LV_CLR_ORANGE           lv_color_hex(0xFFA500)
#define LV_CLR_YELLOW           lv_color_hex(0xFFFF00)
#define LV_CLR_GREEN            lv_color_hex(0x008000)
#define LV_CLR_LIME             lv_color_hex(0x00FF00)
#define LV_CLR_CYAN             lv_color_hex(0x00FFFF)
#define LV_CLR_BLUE             lv_color_hex(0x0000FF)
#define LV_CLR_NAVY             lv_color_hex(0x000080)
#define LV_CLR_MAGENTA          lv_color_hex(0xFF00FF)
#define LV_CLR_PURPLE           lv_color_hex(0x800080)

#define LV_CLR_LIGHT_GRAY       lv_color_hex(0xD3D3D3)
#define LV_CLR_SILVER           lv_color_hex(0xC0C0C0)
#define LV_CLR_GRAY             lv_color_hex(0x808080)
#define LV_CLR_DARK_GRAY        lv_color_hex(0xA9A9A9)
#define LV_CLR_DIM_GRAY         lv_color_hex(0x696969)

#define LV_CLR_OFF_WHITE        lv_color_hex(0xF5F5F5)
#define LV_CLR_SLATE            lv_color_hex(0x708090)
#define LV_CLR_CHARCOAL         lv_color_hex(0x36454F)

#define LV_CLR_PASTEL_RED       lv_color_hex(0xFFB3BA)
#define LV_CLR_PASTEL_PINK      lv_color_hex(0xFFD1DC)
#define LV_CLR_PASTEL_ORANGE    lv_color_hex(0xFFDFBA)
#define LV_CLR_PASTEL_PEACH     lv_color_hex(0xFFE5B4)
#define LV_CLR_PASTEL_YELLOW    lv_color_hex(0xFFFFBA)
#define LV_CLR_PASTEL_GREEN     lv_color_hex(0xBAFFC9)
#define LV_CLR_PASTEL_MINT      lv_color_hex(0xAAF0D1)
#define LV_CLR_PASTEL_BLUE      lv_color_hex(0xBAE1FF)
#define LV_CLR_PASTEL_PURPLE    lv_color_hex(0xD5AAFF)
#define LV_CLR_PASTEL_LILAC     lv_color_hex(0xC8A2C8)

#define LV_CLR_TEAL             lv_color_hex(0x008080)
#define LV_CLR_INDIGO           lv_color_hex(0x4B0082)
#define LV_CLR_BROWN            lv_color_hex(0xA52A2A)
#define LV_CLR_PINK             lv_color_hex(0xFFC0CB)
#define LV_CLR_HOT_PINK         lv_color_hex(0xFF69B4)
#define LV_CLR_GOLD             lv_color_hex(0xFFD700)
#define LV_CLR_CORAL            lv_color_hex(0xFF7F50)
#define LV_CLR_OLIVE            lv_color_hex(0x808000)
#define LV_CLR_AQUAMARINE       lv_color_hex(0x7FFFD4)
#define LV_CLR_LAVENDER         lv_color_hex(0xE6E6FA)
#define LV_CLR_TURQUOISE        lv_color_hex(0x40E0D0)
#define LV_CLR_SALMON           lv_color_hex(0xFA8072)
#define LV_CLR_SKY_BLUE         lv_color_hex(0x87CEEB)
#define LV_CLR_ROYAL_BLUE       lv_color_hex(0x4169E1)
#define LV_CLR_VIOLET           lv_color_hex(0xEE82EE)
#define LV_CLR_CRIMSON          lv_color_hex(0xDC143C)
#define LV_CLR_CREAM            lv_color_hex(0xFFFDD0)

static inline lv_color_t lv_color_to_ghost(lv_color_t color) {
    return lv_color_mix(color, lv_color_white(), 128);
}