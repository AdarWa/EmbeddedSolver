#include "graphic_task.h"

#include <esp_check.h>

#include "draw.h"
#include "esp_log.h"
#include "graphic.h"
#include "freertos/FreeRTOS.h"
#include "lcd/lcd.h"
#include "lcd/touch.h"

static const char* TAG = "task/graphic";

static SemaphoreHandle_t graphic_mutex;

esp_err_t lock_graphic_mutex() {
    if (xSemaphoreTake(graphic_mutex, pdMS_TO_TICKS(20)) != pdPASS) {
        ESP_LOGE(TAG, "Missed graphic mutex lock; sacrificing loop iteration");
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

void unlock_graphic_mutex() {
    xSemaphoreGive(graphic_mutex);
}

static void graphic_task(void* pvParameters) {
    graphic_task_params_t* params = pvParameters;
    esp_lcd_panel_handle_t lcd_handle = params->lcd_handle;
    esp_lcd_touch_handle_t touch_handle = params->touch_handle;

    static esp_lcd_touch_point_data_t touch_data[CONFIG_ESP_LCD_TOUCH_MAX_POINTS];
    static uint8_t point_cnt;


    for (;;) {
        // lock_graphic_mutex();

        esp_lcd_touch_read_data(touch_handle);
        esp_lcd_touch_get_data(touch_handle, touch_data, &point_cnt, CONFIG_ESP_LCD_TOUCH_MAX_POINTS);
        handle_draw(*touch_data, point_cnt);

        draw_buff(lcd_handle);

        // unlock_graphic_mutex();
        vTaskDelay(pdMS_TO_TICKS(GRAPHIC_TASK_LOOP_TIME));
    }
}


esp_err_t configure_lcd(esp_lcd_panel_handle_t* handle) {
    esp_lcd_panel_io_handle_t lcd_io;

    ESP_ERROR_CHECK(lcd_display_brightness_init());

    ESP_ERROR_CHECK(lcd_init(&lcd_io, handle));

    ESP_ERROR_CHECK(lcd_display_brightness_set(50));

    return ESP_OK;
}

esp_err_t configure_touch(esp_lcd_touch_handle_t* handle) {
    return touch_init(handle);
}

esp_err_t start_graphic_task() {
    graphic_mutex = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(graphic_mutex, ESP_ERR_NO_MEM, TAG, "No memory for graphic mutex");

    static esp_lcd_panel_handle_t lcd_handle;
    static esp_lcd_touch_handle_t touch_handle;

    ESP_LOGI(TAG, "Configuring lcd...");

    configure_lcd(&lcd_handle);

    ESP_ERROR_CHECK(configure_touch(&touch_handle));

    ESP_LOGI(TAG, "Starting graphic task...");

    static graphic_task_params_t task_params;
    task_params.lcd_handle = lcd_handle;
    task_params.touch_handle = touch_handle;

    xTaskCreate(graphic_task, GRAPHIC_TASK_NAME, GRAPHIC_STACK_SIZE, &task_params, 1, nullptr);

    return ESP_OK;
}