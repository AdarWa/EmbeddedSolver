//
// Created by adarw on 4/1/26.
//

#pragma once

#include <lvgl.h>
#include "../graphics/screen_manager.h"

lv_obj_t* main_screen() {
    lv_obj_t* root = generate_root(true);
    lv_obj_t* label = lv_label_create(root);
    lv_label_set_text(label, "TEST SCREEN!");
    lv_obj_set_align(label, LV_ALIGN_CENTER);
    return root;
}
