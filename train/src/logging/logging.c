//
// Created by adarw on 4/10/26.
//

#include "logging.h"

static const char* level_names[] = { "DEBUG", "INFO", "WARN", "ERROR", "FATAL" };

static LogLevel min_level = L_DEBUG;

void set_log_level(LogLevel level) {
    min_level = level;
}

void log_msg(LogLevel level, const char *tag, const char *fmt, ...) {
    if (level < min_level) {
        return;
    }
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char time_str[20];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", t);

    printf("[%s] [%s] [%s] ", time_str, level_names[level], tag);

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");

    if (level == L_FATAL) {
        printf("!!! FATAL ERROR: Terminating program !!!\n");
        exit(EXIT_FAILURE);
    }
}