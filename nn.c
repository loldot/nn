#include <stdio.h>
#include <math.h>
#include <time.h>   
#include <stdlib.h>
#include "debug.c"

FILE *mnist_images;
FILE *mnist_labels;
const size_t input_size = 784;

typedef struct Layer
{
    int input_size;
    int output_size;
    float* weights;
    float* bias;
} Layer;


float weights_0[input_size][16];
float bias_0[16];
const Layer layer0 = {input_size, 16, weights_0, bias_0};

// Hidden layer 1
float weights_1[16][16];
float bias_1[16];
const Layer layer1 = {16, 16, weights_1, bias_1};


float weights_2[16][10];
float bias_2[16];
const Layer layer2 = {16, 10, weights_2, bias_2};

const float sigmoid(float x){
    return 1 / (1 + exp(-x));
}

// Forward propagation function
void forward(const Layer* layer, const float* input, float* output) {
    for (int i = 0; i < layer->output_size; i++) {
        float sum = .0f;
        for (int j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[i * layer->input_size + j];
        }
        output[i] = sigmoid(sum + layer->bias[i]);
    }
}

void stocastic_gradient_descent(){
    
}

int recognize_digit(float* input){
    float hidden0[16];
    float hidden1[16];
    float output[10];

    print_image(input);

    forward(&layer0, input, hidden0);
    printf("Hidden 0: ");
    print_tensor(hidden0, 16);
    forward(&layer1, hidden0, hidden1);

    printf("Hidden 1: ");
    print_tensor(hidden1, 16);
    forward(&layer2, hidden1, output);

    printf("Output: ");
    print_tensor(output, 10);
    int max = 0;
    for (size_t i = 0; i < 10; i++)
    {
        max = (output[i] > output[max]) ? i : max;
    }

    return max;
}

void init() {
    srand(time(NULL));
    mnist_images = fopen("./data/train-images.idx3-ubyte", "rb");
    if(mnist_images == NULL){
        printf("Error opening image file\n");
        return;
    }

    mnist_labels = fopen("./data/train-labels.idx1-ubyte", "rb");
    if(mnist_labels == NULL){
        printf("Error opening label file\n");
        return;
    }

    for(int i = 0; i < layer0.input_size; i++){
        for(int j = 0; j < layer0.output_size; j++){
            weights_0[i][j] = ((float)rand() / (float)RAND_MAX) -.5f;
        }
    }
    for (int i = 0; i < layer1.input_size; i++)
    {
        for (int j = 0; j < layer1.output_size; j++)
        {
            weights_1[i][j] = ((float)rand() / (float)RAND_MAX) -.5f;
        }
    }
    for (int i = 0; i < layer2.input_size; i++)
    {
        for (int j = 0; j < layer2.output_size; j++)
        {
            weights_2[i][j] = ((float)rand() / (float)RAND_MAX) -.5f;
        }
    }
}

void fill_input(float* input, int offset){
    fseek(mnist_images, 16 + input_size * offset, SEEK_SET);

    __uint8_t buffer[input_size];
    auto bytesRead = fread(buffer, sizeof(__uint8_t), input_size, mnist_images);
    
    for (size_t i = 0; i < input_size; i++)
    {
        input[i] = buffer[i] / 255.0f;
    }
}

__uint8_t read_label(int offset) {
    fseek(mnist_labels, 8 + offset, SEEK_SET);
    return fgetc(mnist_labels);
}

int main(int n, char** args){
    const int offset = 3;
    float input[input_size];
    
    init();

    fill_input(input, offset);

    auto label = read_label(offset);

    int digit = recognize_digit(&input);

    printf("Label: %d\n", label);
    printf("Guess: %d\n", digit);

    fclose(mnist_images);
    fclose(mnist_labels);
}
