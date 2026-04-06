#include "graphic_task.h"

#include <esp_check.h>

#include "esp_log.h"
#include "graphic.h"
#include "freertos/FreeRTOS.h"
#include "lcd/lcd.h"
#include "lcd/touch.h"

static const char* TAG = "task/graphic";

static SemaphoreHandle_t graphic_mutex;

esp_err_t lock_graphic_mutex() {
    if (xSemaphoreTake(graphic_mutex, 20) != pdPASS) {
        ESP_LOGE(TAG, "Missed graphic mutex lock; sacrificing loop iteration");
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

void unlock_graphic_mutex() {
    xSemaphoreGive(graphic_mutex);
}

static void graphic_task(void* pvParameters) {
    esp_lcd_panel_handle_t* handle = pvParameters;
    for (;;) {
        lock_graphic_mutex();
        draw_buff(*handle);
        unlock_graphic_mutex();
        vTaskDelay(pdMS_TO_TICKS(GRAPHIC_TASK_LOOP_TIME));
    }
}


esp_err_t configure_lcd(esp_lcd_panel_handle_t* handle) {
    esp_lcd_panel_io_handle_t lcd_io;
    esp_lcd_touch_handle_t tp;

    ESP_ERROR_CHECK(lcd_display_brightness_init());

    ESP_ERROR_CHECK(lcd_init(&lcd_io, handle));
    // ESP_ERROR_CHECK(touch_init(&tp));


    ESP_ERROR_CHECK(lcd_display_brightness_set(50));

    return ESP_OK;
}

esp_err_t start_graphic_task() {
    graphic_mutex = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(graphic_mutex, ESP_ERR_NO_MEM, TAG, "No memory for graphic mutex");

    static esp_lcd_panel_handle_t handle;

    ESP_LOGI(TAG, "Configuring lcd...");

    configure_lcd(&handle);

    ESP_LOGI(TAG, "Starting graphic task...");

    xTaskCreate(graphic_task, GRAPHIC_TASK_NAME, GRAPHIC_STACK_SIZE, &handle, 1, nullptr);

    return ESP_OK;
}