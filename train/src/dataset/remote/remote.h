//
// Created by adarw on 4/18/26.
//

#pragma once

#include <stdbool.h>

void remote_fetch_url(const char* url);
void remote_cleanup();
void remote_init();
void remote_fetch_all();
bool remote_all_data_exists();
