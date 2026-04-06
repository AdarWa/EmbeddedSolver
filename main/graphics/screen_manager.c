//
// Created by adarw on 4/1/26.
//

#include "screen_manager.h"

#include <esp_log.h>
#include <esp_lvgl_port.h>
#include <lvgl.h>

#include "colors.h"

#include "lvgl_task.h"

lv_obj_t* generate_root(bool show_back) {
    lv_obj_t* cont = lv_obj_create(nullptr);
    lv_obj_set_size(cont, LV_PCT(100), LV_PCT(100));
    lv_obj_set_style_bg_color(cont, LV_CLR_CREAM, 0);

    if (show_back) {
        lv_obj_t* back_btn = lv_btn_create(cont);
        lv_obj_set_size(back_btn, 30, 30);
        lv_obj_set_style_bg_color(back_btn, LV_CLR_AQUAMARINE, 0);

        lv_obj_align(back_btn, LV_ALIGN_TOP_LEFT, 10, 10);

        lv_obj_t* label = lv_label_create(back_btn);
        lv_label_set_text(label, LV_SYMBOL_LEFT);
        lv_obj_set_style_text_color(label, LV_CLR_BLACK, LV_PART_MAIN | LV_STATE_DEFAULT);
        lv_obj_center(label);
        lv_obj_update_layout(back_btn);
        ESP_LOGI("screen_manager", "x: %d, y: %d", lv_obj_get_x(back_btn),lv_obj_get_y(back_btn));
    }
    return cont;
}

void move_to_screen(lv_obj_t* defined_screen) {
    ESP_ERROR_CHECK(lock_lvgl_mutex());
    lv_scr_load_anim(defined_screen, LV_SCR_LOAD_ANIM_MOVE_LEFT, NEW_SCR_ANIMATION_DELAY, 0, true);
    unlock_lvgl_mutex();
}
