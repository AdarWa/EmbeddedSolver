file(GLOB_RECURSE component_srcs "*.c")

idf_component_register(SRCS "${component_srcs}"
        PRIV_REQUIRES spi_flash
        REQUIRES esp_lcd_touch_xpt2046 esp_lcd_ili9341 esp_driver_ledc esp_timer
        INCLUDE_DIRS ".")