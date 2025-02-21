#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "debug.c"
#include "io.c"

#define input_size 784
#define hidden_size 16
#define output_size 10

#define epochs 30
#define traing_set_size 60000
#define learning_rate 0.01

const int max_index(const int size, const double tensor[size]){
    int max = 0;
    for (size_t i = 0; i < size; i++)
    {
        max = (tensor[i] > tensor[max]) ? i : max;
    }
    return max;
}

__uint8_t* data = NULL;
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

// derivative of tanh = (1-tanh²(x))
const double activation_prime(double tanh){
    return (1 - tanh) * (1 + tanh);
}

// Forward propagation function
void forward(
    const int m,
    const double input[m],
    const int n,  
    double output[n],
    const double weights[m][n],
    const double bias[n]
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
    const int m,
    const double input[m],
    const int n,  
    double output[n],
    const double weights[m][n],
    const double bias[n]
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

void predict(
    double input[input_size], 
    double hidden[hidden_size], 
    double hidden2[hidden_size], 
    double output[output_size]){
    forward(
        input_size, input, 
        hidden_size, hidden, 
        weights_0, bias_0
    );
    forward(
        hidden_size, hidden, 
        hidden_size, hidden2, 
        weights_1, bias_1
    );
    forward_softmax(
        hidden_size, hidden2, 
        output_size, output,
        weights_2, bias_2
    );
}

void predict_debug(
    double input[input_size], 
    double hidden[hidden_size], 
    double hidden2[hidden_size], 
    double output[output_size]){
    forward(
        input_size, input,  
        hidden_size, hidden, 
        weights_0, bias_0
    );
    printf("Hidden 0: ");
    print_tensor(hidden, hidden_size);

    printf("Bias: ");
    print_tensor(bias_0, hidden_size);

    forward(
        hidden_size, hidden, 
        hidden_size, hidden2, 
        weights_1, bias_1
    );
    
    printf("Hidden 1: ");
    print_tensor(hidden2, hidden_size);

    printf("Bias: ");
    print_tensor(bias_1, hidden_size);

    forward_softmax(
        hidden_size, hidden2,  
        output_size, output,
        weights_2, bias_2
    );
    printf("Output: ");
    highlight_tensor(output, output_size, max_index(output_size, output));
}


int recognize_digit(const double input[input_size]){
    double hidden[hidden_size];
    double hidden2[hidden_size];
    double output[output_size];

    print_image(input);
    predict_debug(input, hidden, hidden2, output);

    return max_index(output_size, output);
}

unsigned char backprop(const double input[input_size], const double expected[output_size]){
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

    return max_index(output_size, output);
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

            fill_input(data, input_size, input, e);
            int guess = recognize_digit(input);
            printf("Guess: %d\n", guess);

        for (size_t i = 0; i < traing_set_size; i++)
        {
            int offset = rand() % traing_set_size;

            fill_input(data, input_size, input, offset);

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
    data = open_dataset(mnist_images);
    // FILE* model = fopen("model.bin", "rb");
    // if(model != NULL){
    //     load_model(weights_0, bias_0, weights_1, bias_1, "model.bin");
    //     fclose(model);
    //     return;
    // }

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
    init();
    stocastic_gradient_descent(); 

    munmap(data, traing_set_size);
    fclose(mnist_labels);
}
