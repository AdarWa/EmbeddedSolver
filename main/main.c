#include "esp_flash.h"
#include "esp_log.h"
#include "graphics/lvgl_task.h"
#include "utils/chip_info.h"
#include "screens/defined_screens.h"

void start_all_tasks() {
    ESP_ERROR_CHECK(startLvglTask());
}

void set_default_screen() {
    move_to_screen(main_screen());
}

void app_main(void){
    print_info();
    start_all_tasks();
    set_default_screen();
}
