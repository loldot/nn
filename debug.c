#include <stdio.h>

void print_image(float* input){
    for (auto i = 0; i < 784; i++)
    {
        if(i % 28 == 0){
            putchar('\n');
        }
        printf("%3d", (int)(255 * input[i]));
    }
    putchar('\n');
}

void print_tensor(float* tensor, int size){
    for (auto i = 0; i < size; i++)
    {
        printf("%.3f ", tensor[i]);
    }
    putchar('\n');
}