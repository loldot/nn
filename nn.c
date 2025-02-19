#include <stdio.h>
#include <math.h>
#include <time.h>   
#include <stdlib.h>
#include "debug.c"
#include "nn.h"
#include "io.c"

FILE *mnist_images;
FILE *mnist_labels;

// Input layer
double weights_0[input_size][hidden_size];
double bias_0[hidden_size];

// Hidden layer 1
double weights_1[hidden_size][hidden_size];
double bias_1[hidden_size];

// Hidden layer 2
double weights_2[hidden_size][output_size];
double bias_2[output_size];

const double activation(double x){
    return tanh(x);
}
const double activation_prime(double x){
    return (1 - x) * (1 + x);
}

// Forward propagation function
void forward(
    const double* input, int m, 
    double* output, int n,
    double weights[m][n],
    double* bias
) {
    for (int i = 0; i < n; i++) {
        double sum = .0f;
        for (int j = 0; j < m; j++) {
            sum += input[j] * weights[j][i];
        }
        output[i] = activation(sum + bias[i]);
    }
}


void forward_softmax(
    const double* input, int m, 
    double* output, int n,
    double weights[m][n], 
    double* bias
) {
    double total = 0.0f;
    for (int i = 0; i < n; i++) {
        double sum = .0f;
        
        for (int j = 0; j < m; j++) {
            sum += input[j] * weights[j][i];
        }
        output[i] = exp(sum + bias[i]);
        total += output[i];
    }
    
    for (int i = 0; i < n; i++) {
        output[i] /= total;
    }
}

void predict(double* input, double* hidden, double* hidden2, double* output){
    forward(
        input, input_size, 
        hidden, hidden_size,
        weights_0, bias_0
    );
    forward(
        hidden, hidden_size, 
        hidden2, hidden_size,
        weights_1, bias_1
    );
    forward_softmax(
        hidden2, hidden_size, 
        output, output_size,
        weights_2, bias_2
    );
}

void predict_debug(double* input, double* hidden, double* hidden2, double* output){
    forward(
        input, input_size, 
        hidden, hidden_size,
        weights_0, bias_0
    );
    printf("Hidden 0: ");
    print_tensor(hidden, hidden_size);

    printf("Bias: ");
    print_tensor(bias_0, hidden_size);

    forward(
        hidden, hidden_size, 
        hidden2, hidden_size,
        weights_1, bias_1
    );
    
    printf("Hidden 1: ");
    print_tensor(hidden2, hidden_size);

    printf("Bias: ");
    print_tensor(bias_1, hidden_size);

    forward_softmax(
        hidden2, hidden_size, 
        output, output_size,
        weights_2, bias_2
    );
    printf("Output: ");
    highlight_tensor(output, output_size, max_index(output, output_size));
}


int recognize_digit(double* input){
    double hidden[hidden_size];
    double hidden2[hidden_size];
    double output[output_size];

    print_image(input);
    predict_debug(input, hidden, hidden2, output);

    return max_index(output, output_size);
}

unsigned char backprop(const double* input, const double* expected){
    double hidden0[hidden_size];
    double hidden1[hidden_size];
    double output[output_size];
    double output_error[output_size];
    double hidden0_error[hidden_size];
    double hidden1_error[hidden_size];


    predict(input, hidden0, hidden1, output);

    // 1. Calculate Output Layer Error
    for (size_t i = 0; i < output_size; i++)
    {
        double derivative = (1 - output[i]) * output[i];
        output_error[i] = (expected[i] - output[i]) * derivative;
    }

    // 2. Calculate Hidden Layer 1 Error
    for (int i = 0; i < hidden_size; i++) {
        double sum = 0.0f;
        for (int j = 0; j < output_size; j++) {
            sum += weights_2[i][j] * output_error[j];
        }
        hidden1_error[i] = sum * activation_prime(hidden1[i]);
    }

    // 3. Calculate Hidden Layer 2 Error
    for (int i = 0; i < hidden_size; i++) {
        double sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += weights_1[i][j] * hidden1_error[j];
        }
        hidden0_error[i] = sum * activation_prime(hidden0[i]);
    }

    // 3. Update Hidden Layer Weights and Biases
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights_0[j][i] += learning_rate *  hidden0_error[i] * input[j];
        }
        bias_0[i] += learning_rate * hidden0_error[i];
    }

    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            weights_1[j][i] += learning_rate *  hidden1_error[i] * hidden0[j];
        }
        bias_1[i] += learning_rate * hidden1_error[i];
    }

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            weights_2[j][i] += learning_rate * output_error[i] * hidden1[j];
        }
        bias_2[i] += learning_rate * output_error[i];
    }

    return max_index(output, output_size);
}

void stocastic_gradient_descent(){
    double input[input_size];
    double hidden[hidden_size];
    double output[output_size];
    
    double label[output_size];
    unsigned char expected;
    unsigned char guess;

    int correct = 0;

    for (size_t e = 0; e < epochs; e++)
    {
        correct = 0;

            fill_input(mnist_images, input, e);
            int guess = recognize_digit(input);
            printf("Guess: %d\n", guess);

        for (size_t i = 0; i < traing_set_size; i++)
        {
            int offset = rand() % traing_set_size;

            fill_input(mnist_images, input, offset);
            expected = read_label(mnist_labels, offset);

            for (size_t i = 0; i < output_size; i++)
            {
                label[i] = 0.0f;
            }
            label[expected] = 1.0f;

            guess = backprop(input, label);
            correct += (expected == guess) ? 1 : 0;
        }        
        printf("Epoch %d: %d / %d\n", e, correct, traing_set_size);

    }
}

void init() {
    srand(0xdeadbeef);
    mnist_images = fopen("./data/train-images.idx3-ubyte", "rb");
    mnist_labels = fopen("./data/train-labels.idx1-ubyte", "rb");

    if(mnist_images == NULL){
        printf("Error opening image file\n");
        return;
    }

    if(mnist_labels == NULL){
        printf("Error opening label file\n");
        return;
    }

    FILE* model = fopen("model.bin", "rb");
    if(model != NULL){
        load_model(weights_0, bias_0, weights_1, bias_1, "model.bin");
        fclose(model);
        return;
    }

    for(int i = 0; i < hidden_size; i++){
        for(int j = 0; j < input_size; j++){
            weights_0[j][i] = (double)(rand() % 100) / 10000.0f;
        }
        bias_0[i] = 0;
    }
    for (int i = 0; i < hidden_size; i++)
    {
        for (int j = 0; j < hidden_size ; j++)
        {
            weights_1[i][j] = (double)(rand() % 100) / 10000.0f;
        }
        bias_1[i] = 0;
    }

    for (int i = 0; i < output_size; i++)
    {
        for (int j = 0; j < hidden_size; j++)
        {
            weights_2[i][j] = (double)(rand() % 100) / 1000.0f;
        }
        bias_2[i] = 0;
    }
}

int main(int n, char** args){
    const int offset = 103;
    double input[input_size];
    
    init();

    // fill_input(mnist_images, input, offset);

    // int label = read_label(mnist_labels, offset);
    // int digit = recognize_digit(&input);

    // printf("Label: %d\n", label);
    // printf("Guess: %d\n", digit);

    stocastic_gradient_descent();

    // store_model(weights_0, bias_0, weights_1, bias_1, "model.bin");
    
    fclose(mnist_images);
    fclose(mnist_labels);
}
