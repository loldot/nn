#include <stdio.h>

void print_image(const float* input){
    for (int i = 0; i < 784; i++)
    {
        if(i % 28 == 0){
            putchar('\n');
        }
        printf("%3d", (int)(255 * input[i]));
    }
    putchar('\n');
}

void highlight_tensor(const float* tensor, int size, int highlight){
    for (int i = 0; i < size; i++)
    {
        if(i == highlight){
            printf("\033[1;32m");
        }
        printf("%.3f ", tensor[i]);
        if(i == highlight){
            printf("\033[0m");
        }
    }
    putchar('\n');
}

void print_tensor(float* tensor, int size) {
    highlight_tensor(tensor, size, -1);
}