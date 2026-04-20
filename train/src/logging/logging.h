//
// Created by adarw on 4/10/26.
//

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>

typedef enum {
    L_DEBUG,
    L_INFO,
    L_WARN,
    L_ERROR,
    L_FATAL
} LogLevel;

void set_log_level(LogLevel level);
void log_msg(LogLevel level, const char *tag, const char *fmt, ...);

#define LOG_D(tag, ...) log_msg(L_DEBUG, tag, __VA_ARGS__)
#define LOG_I(tag, ...) log_msg(L_INFO,  tag, __VA_ARGS__)
#define LOG_W(tag, ...) log_msg(L_WARN,  tag, __VA_ARGS__)
#define LOG_E(tag, ...) log_msg(L_ERROR, tag, __VA_ARGS__)
#define LOG_F(tag, ...) log_msg(L_FATAL, tag, __VA_ARGS__)