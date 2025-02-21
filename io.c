#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>

__uint8_t read_label(FILE* mnist_labels, int offset) {
    fseek(mnist_labels, 8 + offset, SEEK_SET);
    return fgetc(mnist_labels);
}

__uint8_t* open_dataset(FILE* file)
{
    struct stat st;
    fstat(file->_fileno, &st);

    __uint8_t* addr = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, file->_fileno, 0);
    fclose(file);

    return addr;
}

void fill_input(__uint8_t* addr, int n, double input[n], int offset){
    for (size_t i = 0; i < n; i++)
    {
        input[i] = (double)(addr[16 + (n * offset) + i]) / 255.0;
    }
}

// void store_model(
//     const double* weights, 
//     const double* bias, 
//     const double* hidden_weights,
//     const double* hidden_bias,
//     const char* filename){
//     FILE* file = fopen(filename, "wb");
//     fwrite(weights, sizeof(double), input_size * 16, file);
//     fwrite(bias, sizeof(double), 16, file);
//     fwrite(hidden_weights, sizeof(double), 16 * 10, file);
//     fwrite(hidden_bias, sizeof(double), 10, file);
//     fclose(file);
// }

// void load_model(
//     double* weights, 
//     double* bias, 
//     double* hidden_weights,
//     double* hidden_bias,
//     const char* filename){
//     FILE* file = fopen(filename, "rb");
//     fread(weights, sizeof(double), input_size * 16, file);
//     fread(bias, sizeof(double), 16, file);
//     fread(hidden_weights, sizeof(double), 16 * 10, file);
//     fread(hidden_bias, sizeof(double), 10, file);
//     fclose(file);
// }