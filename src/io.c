#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>

unsigned char* read_labels(const char* filename) {
    unsigned int header[2];
    unsigned int rows, cols;
    unsigned char* labels;

    FILE* mnist_labels = fopen(filename, "rb");
    
    if(mnist_labels == NULL){
        perror("Failed to open file");
        return NULL;
    }

    fread(header, sizeof(__uint32_t), 2, mnist_labels);
    rows = __builtin_bswap32(header[1]);
    printf("Rows: %d\n", rows);

    labels = malloc(rows);
    fread(labels, 1, rows, mnist_labels);    
    fclose(mnist_labels);

    return labels;
}

unsigned char* open_dataset(const char* filename, struct stat *data_stat)
{
    FILE* file = fopen(filename, "rb");

    if(file == NULL || fstat(file->_fileno, data_stat) > 0){
        perror("Failed to open file");
        return NULL;
    }

    printf("File size: %ld\n", data_stat->st_size);

    unsigned char* addr = mmap(0, data_stat->st_size, PROT_READ, MAP_PRIVATE, file->_fileno, 0);
    fclose(file);

    return addr;
}

void fill_input(__uint8_t* addr, int n, float input[n], int offset){
    for (size_t i = 0; i < n; i++)
    {
        input[i] = (float)(addr[16 + (n * offset) + i]) / 255.0f;
    }
}

// void store_model(
//     const float* weights, 
//     const float* bias, 
//     const float* hidden_weights,
//     const float* hidden_bias,
//     const char* filename){
//     FILE* file = fopen(filename, "wb");
//     fwrite(weights, sizeof(float), input_size * 16, file);
//     fwrite(bias, sizeof(float), 16, file);
//     fwrite(hidden_weights, sizeof(float), 16 * 10, file);
//     fwrite(hidden_bias, sizeof(float), 10, file);
//     fclose(file);
// }

// void load_model(
//     float* weights, 
//     float* bias, 
//     float* hidden_weights,
//     float* hidden_bias,
//     const char* filename){
//     FILE* file = fopen(filename, "rb");
//     fread(weights, sizeof(float), input_size * 16, file);
//     fread(bias, sizeof(float), 16, file);
//     fread(hidden_weights, sizeof(float), 16 * 10, file);
//     fread(hidden_bias, sizeof(float), 10, file);
//     fclose(file);
// }