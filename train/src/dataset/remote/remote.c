//
// Created by adarw on 4/18/26.
//

#include "remote.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <zlib.h>
#include <stdbool.h>

#include "logging/logging.h"

#define CHUNK_SIZE 16384

static const char* TAG = "remote";
static const char* TRAIN_IMG_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz";
static const char* TRAIN_LABEL_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz";
static const char* TEST_IMG_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz";
static const char* TEST_LABEL_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz";

static CURL* curl = NULL;

static char* extract_filename(const char* url) {
    if (url == NULL) {
        return NULL;
    }

    const char *end = url;
    while (*end != '\0' && *end != '?' && *end != '#') {
        end++;
    }

    const char *last_slash = NULL;
    const char *ptr = url;
    while (ptr < end) {
        if (*ptr == '/') {
            last_slash = ptr;
        }
        ptr++;
    }

    const char *start;
    if (last_slash != NULL) {
        start = last_slash + 1;
    } else {
        start = url;
    }

    size_t length = end - start;

    char *filename = (char*)malloc(length + 1);
    if (filename == NULL) {
        return NULL;
    }

    strncpy(filename, start, length);
    filename[length] = '\0';

    return filename;
}

static char* create_extracted_filename(const char* gz_filename) {
    size_t len = strlen(gz_filename);

    if (len > 3 && strcmp(gz_filename + len - 3, ".gz") == 0) {
        char* ext_filename = (char*)malloc(len - 2); // len - 3 + 1 for \0
        if (ext_filename) {
            strncpy(ext_filename, gz_filename, len - 3);
            ext_filename[len - 3] = '\0';
            return ext_filename;
        }
    }

    char* ext_filename = (char*)malloc(len + 11);
    if (ext_filename) {
        sprintf(ext_filename, "%s.extracted", gz_filename);
    }
    return ext_filename;
}

static size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

static int fetch_file(const char* url, const char* filename) {
    if (!curl) {
        LOG_E(TAG, "Error: CURL not initialized");
        return 0;
    }

    FILE *fp = fopen(filename, "wb");
    if(!fp) {
        LOG_E(TAG, "Error: Failed to open %s for writing", filename);
        return 0;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    LOG_D(TAG, "Downloading %s...", url);

    CURLcode res = curl_easy_perform(curl);
    fclose(fp);

    if(res != CURLE_OK) {
        LOG_E(TAG,"curl_easy_perform() failed: %s", curl_easy_strerror(res));
        return 0;
    }

    LOG_I(TAG, "Successfully downloaded to %s", filename);
    return 1;
}

/* Helper to handle the zlib extraction */
static int extract_gzip(const char *source_gz, const char *dest_file) {
    gzFile infile = gzopen(source_gz, "rb");
    if (!infile) {
        LOG_E(TAG, "Error: Failed to open gzip file %s", source_gz);
        return 0;
    }

    FILE *outfile = fopen(dest_file, "wb");
    if (!outfile) {
        LOG_E(TAG, "Error: Failed to open output file %s", dest_file);
        gzclose(infile);
        return 0;
    }

    char buffer[CHUNK_SIZE];
    int uncompressed_bytes;

    LOG_D(TAG, "Extracting %s to %s...", source_gz, dest_file);

    while ((uncompressed_bytes = gzread(infile, buffer, CHUNK_SIZE)) > 0) {
        size_t written = fwrite(buffer, 1, uncompressed_bytes, outfile);
        if (written != uncompressed_bytes) {
             LOG_E(TAG, "Error: Failed writing data to %s", dest_file);
             fclose(outfile);
             gzclose(infile);
             return 0;
        }
    }

    if (uncompressed_bytes < 0) {
        int err;
        LOG_E(TAG, "zlib Error: %s", gzerror(infile, &err));
        fclose(outfile);
        gzclose(infile);
        return 0;
    }

    fclose(outfile);
    gzclose(infile);
    LOG_D(TAG, "Successfully extracted to %s", dest_file);
    return 1;
}

void remote_init() {
    LOG_I(TAG, "Initializing remote fetching...");
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
}

void remote_cleanup() {
    LOG_I(TAG, "Cleaning up remote...");
    if (curl) {
        curl_easy_cleanup(curl);
        curl = NULL;
    }
    curl_global_cleanup();
}

void remote_fetch_url(const char* url) {
    LOG_D(TAG, "Attempting to fetch remote %s", url);
    char* gz_filename = extract_filename(url);

    if (!gz_filename) {
        LOG_E(TAG, "Error: Failed to extract filename from URL");
        return;
    }

    if (fetch_file(url, gz_filename)) {
        char* ext_filename = create_extracted_filename(gz_filename);
        if (ext_filename) {
            extract_gzip(gz_filename, ext_filename);
            free(ext_filename);
        }
    }

    free(gz_filename);
}

void remote_fetch_all() {
    remote_fetch_url(TRAIN_IMG_URL);
    remote_fetch_url(TRAIN_LABEL_URL);
    remote_fetch_url(TEST_IMG_URL);
    remote_fetch_url(TEST_LABEL_URL);
}

static bool file_exists(const char* filename) {
    if (filename == NULL) {
        return false;
    }

    FILE* file = fopen(filename, "rb");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

bool remote_all_data_exists() {
    const char* train_img_file = "train-images-idx3-ubyte";
    const char* train_lbl_file = "train-labels-idx1-ubyte";
    const char* test_img_file  = "t10k-images-idx3-ubyte";
    const char* test_lbl_file  = "t10k-labels-idx1-ubyte";

    LOG_D(TAG, "Checking if all MNIST dataset files exist locally...");

    bool exists = file_exists(train_img_file) &&
                  file_exists(train_lbl_file) &&
                  file_exists(test_img_file) &&
                  file_exists(test_lbl_file);

    if (exists) {
        LOG_I(TAG, "All dataset files found on disk.");
    } else {
        LOG_I(TAG, "One or more dataset files are missing.");
    }

    return exists;
}