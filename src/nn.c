#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/mman.h>
#include "debug.h"
#include "io.h"
#include "nn.h"

#define input_size 784
#define hidden_size 16
#define output_size 10

#define epochs 30
#define training_set_size 60000

float learning_rate = 0.01f;

int max_index(const int size, const float tensor[size])
{
    int max = 0;
    for (int i = 0; i < size; i++)
    {
        max = (tensor[i] > tensor[max]) ? i : max;
    }
    return max;
}

struct stat data_stat;
unsigned char *data = NULL;
unsigned char *labels = NULL;

// Input layer
float weights_0[input_size][hidden_size];
float bias_0[hidden_size];

// Hidden layer 1
float weights_1[hidden_size][hidden_size];
float bias_1[hidden_size];

// Hidden layer 2
float weights_2[hidden_size][output_size];
float bias_2[output_size];

float activation(float x)
{
    return tanh(x);
}

// derivative of tanh = (1-tanh²(x))
float activation_prime(float tanh)
{
    return (1 - tanh) * (1 + tanh);
}

// Feed forward
void forward(
    const int m,
    const float input[m],
    const int n,
    float output[n],
    const float weights[m][n],
    const float bias[n])
{
    for (int i = 0; i < n; i++)
    {
        float sum = .0f;
        for (int j = 0; j < m; j++)
        {
            sum += input[j] * weights[j][i];
        }
        output[i] = activation(sum + bias[i]);
    }
}

void forward_linear(
    const int m,
    const float input[m],
    const int n,
    float output[n],
    const float weights[m][n],
    const float bias[n])
{
    for (int i = 0; i < n; i++)
    {
        float sum = .0f;
        for (int j = 0; j < m; j++)
        {
            sum += input[j] * weights[j][i];  // Fixed: weights[input][output]
        }
        output[i] = sum + bias[i];
    }
}

void softmax(const int m, float input[m])
{
    // Find max for numerical stability
    float max = input[0];
    for (int i = 1; i < m; i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < m; i++)
    {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }

    for (int i = 0; i < m; i++)
    {
        input[i] = input[i] / sum;
    }
}

void predict(
    const float input[input_size],
    float hidden[hidden_size],
    float hidden2[hidden_size],
    float output[output_size])
{
    forward(
        input_size, input,
        hidden_size, hidden,
        weights_0, bias_0);
    forward(
        hidden_size, hidden,
        hidden_size, hidden2,
        weights_1, bias_1);
    forward_linear(
        hidden_size, hidden2,
        output_size, output,
        weights_2, bias_2);
    softmax(output_size, output);
}

void predict_debug(
    const float input[input_size],
    float hidden[hidden_size],
    float hidden2[hidden_size],
    float output[output_size])
{
    forward(
        input_size, input,
        hidden_size, hidden,
        weights_0, bias_0);
    printf("Hidden 0: ");
    print_tensor(hidden, hidden_size);

    printf("Bias: ");
    print_tensor(bias_0, hidden_size);

    forward(
        hidden_size, hidden,
        hidden_size, hidden2,
        weights_1, bias_1);

    printf("Hidden 1: ");
    print_tensor(hidden2, hidden_size);

    printf("Bias: ");
    print_tensor(bias_1, hidden_size);

    forward_linear(
        hidden_size, hidden2,
        output_size, output,
        weights_2, bias_2);
    softmax(output_size, output);

    printf("Output: ");
    highlight_tensor(output, output_size, max_index(output_size, output));
}

int recognize_digit(const float input[input_size])
{
    float hidden[hidden_size];
    float hidden2[hidden_size];
    float output[output_size];

    print_image(input);
    predict_debug(input, hidden, hidden2, output);

    return max_index(output_size, output);
}

int backprop(const float input[input_size], const float expected[output_size])
{
    float output[output_size];
    float hidden1[hidden_size];
    float hidden2[hidden_size];
    
    float output_error[output_size];
    float hidden1_error[hidden_size];
    float hidden2_error[hidden_size];

    predict(input, hidden1, hidden2, output);

    // 1. Calculate Output Layer Error (MSE with softmax)
    for (size_t i = 0; i < output_size; i++)
    {
        output_error[i] = expected[i] - output[i];
    }

    // 2. Calculate Hidden Layer 2 Error
    for (int i = 0; i < hidden_size; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < output_size; j++)
        {
            sum += weights_2[i][j] * output_error[j];  // Fixed: weights[input][output]
        }
        hidden2_error[i] = sum * activation_prime(hidden2[i]);
    }

    // 3. Calculate Hidden Layer 1 Error
    for (size_t i = 0; i < hidden_size; i++)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < hidden_size; j++)
        {
            sum += weights_1[i][j] * hidden2_error[j];  // Fixed: weights[input][output]
        }
        hidden1_error[i] = sum * activation_prime(hidden1[i]);
    }   

    // 4. Update Weights and Biases
    // Update weights_0: input -> hidden1
    for (size_t j = 0; j < input_size; j++)
    {
        for (size_t i = 0; i < hidden_size; i++)
        {
            weights_0[j][i] += learning_rate * hidden1_error[i] * input[j];
        }
    }
    for (size_t i = 0; i < hidden_size; i++)
    {
        bias_0[i] += learning_rate * hidden1_error[i];
    }

    // Update weights_1: hidden1 -> hidden2
    for (size_t j = 0; j < hidden_size; j++)
    {
        for (size_t i = 0; i < hidden_size; i++)
        {
            weights_1[j][i] += learning_rate * hidden2_error[i] * hidden1[j];
        }
    }
    for (size_t i = 0; i < hidden_size; i++)
    {
        bias_1[i] += learning_rate * hidden2_error[i];
    }

    // Update weights_2: hidden2 -> output
    for (size_t j = 0; j < hidden_size; j++)
    {
        for (size_t i = 0; i < output_size; i++)
        {
            weights_2[j][i] += learning_rate * output_error[i] * hidden2[j];
        }
    }
    for (size_t i = 0; i < output_size; i++)
    {
        bias_2[i] += learning_rate * output_error[i];
    }

    return max_index(output_size, output);
}

void stochastic_gradient_descent()
{
    float input[input_size];
    float label[output_size];

    int expected, guess;
    int correct = 0;

    for (size_t e = 0; e < epochs; e++)
    {
        correct = 0;

        fill_input(data, input_size, input, e);
        guess = recognize_digit(input);
        printf("Guess: %d\n", guess);

        for (size_t i = 0; i < training_set_size; i++)
        {
            int offset = rand() % training_set_size;
            fill_input(data, input_size, input, offset);

            expected = labels[offset];

            for (size_t i = 0; i < output_size; i++)
            {
                label[i] = 0.0f;
            }
            label[expected] = 1.0f;

            guess = backprop(input, label);
            correct += (expected == guess) ? 1 : 0;
        }
        learning_rate *= 0.9f; // Decay learning rate
        printf("Epoch %zu: %d / %d\n", e, correct, training_set_size);
    }
}

void init()
{
    srand(0xdeadbeef);
    labels = read_labels("./data/train-labels.idx1-ubyte");
    data = open_dataset("./data/train-images.idx3-ubyte", &data_stat);

    if (labels == NULL || data == NULL)
        exit(1);

    // Xavier/Glorot initialization for better convergence
    for (int j = 0; j < input_size; j++)
    {
        for (int i = 0; i < hidden_size; i++)
        {
            weights_0[j][i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrt(6.0f / (input_size + hidden_size));
        }
    }
    for (int i = 0; i < hidden_size; i++)
    {
        bias_0[i] = 0.0f;
    }
    
    for (int j = 0; j < hidden_size; j++)
    {
        for (int i = 0; i < hidden_size; i++)
        {
            weights_1[j][i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrt(6.0f / (hidden_size + hidden_size));
        }
    }
    for (int i = 0; i < hidden_size; i++)
    {
        bias_1[i] = 0.0f;
    }

    for (int j = 0; j < hidden_size; j++)
    {
        for (int i = 0; i < output_size; i++)
        {
            weights_2[j][i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrt(6.0f / (hidden_size + output_size));
        }
    }
    for (int i = 0; i < output_size; i++)
    {
        bias_2[i] = 0.0f;
    }
}

int run_training(int argc, char const *argv[])
{
    init();
    stochastic_gradient_descent();

    munmap(data, data_stat.st_size);
    free(labels);

    return 0;
}
