#include <graphics/graphic.h>

#include "esp_flash.h"
#include "esp_log.h"
#include "graphics/graphic_task.h"
#include "utils/chip_info.h"

void start_all_tasks() {
    ESP_ERROR_CHECK(start_graphic_task());
}

void app_main(void){
    print_info();
    allocate_drawing_buff();
    clear_draw_buff();
    start_all_tasks();
}
