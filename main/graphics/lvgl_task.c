#include "lvgl_task.h"

#include <esp_check.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "lcd/lcd.h"
#include "lcd/touch.h"

static const char* TAG = "task/lvgl";

esp_err_t lock_lvgl_mutex() {
    if (!lvgl_port_lock(300)) {
        ESP_LOGE(TAG, "Missed lvgl port lock; sacrificing loop iteration");
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

void unlock_lvgl_mutex() {
    lvgl_port_unlock();
}

esp_err_t configureLvgl() {
    esp_lcd_panel_io_handle_t lcd_io;
    esp_lcd_panel_handle_t lcd_panel;
    esp_lcd_touch_handle_t tp;
    lvgl_port_touch_cfg_t touch_cfg = {0};
    lv_display_t* lvgl_display = nullptr;

    ESP_ERROR_CHECK(lcd_display_brightness_init());

    ESP_ERROR_CHECK(app_lcd_init(&lcd_io, &lcd_panel));
    lvgl_display = app_lvgl_init(lcd_io, lcd_panel);
    if (lvgl_display == NULL)
    {
        ESP_RETURN_ON_ERROR(ESP_FAIL, TAG, "fatal error in app_lvgl_init");
    }
    ESP_ERROR_CHECK(touch_init(&tp));
    touch_cfg.disp = lvgl_display;
    touch_cfg.handle = tp;
    lv_indev_t *touch_indev = lvgl_port_add_touch(&touch_cfg);
    if (touch_indev) {
        lv_indev_set_mode(touch_indev, LV_INDEV_MODE_TIMER);
    }

    ESP_ERROR_CHECK(lcd_display_brightness_set(50));
    ESP_ERROR_CHECK(lcd_display_rotate(lvgl_display, LV_DISPLAY_ROTATION_90));

    return ESP_OK;
}

esp_err_t startLvglTask() {
    configureLvgl();
    return ESP_OK;
}