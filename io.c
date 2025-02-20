#include <stdio.h>
#include <stdlib.h>

__uint8_t read_label(FILE* mnist_labels, int offset) {
    fseek(mnist_labels, 8 + offset, SEEK_SET);
    return fgetc(mnist_labels);
}

void fill_input(FILE* mnist_images, int n, double input[n], int offset){
    fseek(mnist_images, 16 + n * offset, SEEK_SET);

    __uint8_t buffer[n];
    auto bytesRead = fread(buffer, sizeof(__uint8_t), n, mnist_images);
    
    for (size_t i = 0; i < n; i++)
    {
        input[i] = (double)buffer[i] / 255.0;
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