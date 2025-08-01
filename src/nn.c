#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>

#include "debug.h"
#include "nn.h"
#include "io.h"

#define input_size 784
#define hidden_size 32
#define output_size 10

#define epochs 30
#define training_set_size 60000

float learning_rate = 0.01f;

struct stat data_stat;
unsigned char *data = NULL;
unsigned char *labels = NULL;

// Input layer
float weights_0[hidden_size][input_size];
float bias_0[hidden_size];

// Hidden layer 1
float weights_1[hidden_size][hidden_size];
float bias_1[hidden_size];

// Hidden layer 2
float weights_2[output_size][hidden_size];
float bias_2[output_size];

int max_index(const int size, const float tensor[size])
{
    int max = 0;
    for (int i = 0; i < size; i++)
    {
        max = (tensor[i] > tensor[max]) ? i : max;
    }
    return max;
}

float xavier_init(float m, float n)
{
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sqrt(6.0f / (m + n));
}

void init_layer(
    const int m,
    const int n,
    float weights[n][m],
    float bias[n])
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            weights[i][j] = xavier_init(m, n);
        }
    }
    for (int i = 0; i < n; i++)
    {
        bias[i] = 0.0f;
    }
}

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
    const float weights[n][m],
    const float bias[n])
{
    for (int i = 0; i < n; i++)
    {
        float sum = .0f;
        for (int j = 0; j < m; j++)
        {
            sum += input[j] * weights[i][j];
        }
        output[i] = activation(sum + bias[i]);
    }
}

void forward_linear(
    const int m,
    const float input[m],
    const int n,
    float output[n],
    const float weights[n][m],
    const float bias[n])
{
    for (int i = 0; i < n; i++)
    {
        float sum = .0f;
        for (int j = 0; j < m; j++)
        {
            sum += input[j] * weights[i][j];
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
            sum += weights_2[j][i] * output_error[j];
        }
        hidden2_error[i] = sum * activation_prime(hidden2[i]);
    }

    // 3. Calculate Hidden Layer 1 Error
    for (size_t i = 0; i < hidden_size; i++)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < hidden_size; j++)
        {
            sum += weights_1[j][i] * hidden2_error[j];
        }
        hidden1_error[i] = sum * activation_prime(hidden1[i]);
    }

    // 4. Update Weights and Biases
    // Update weights_0: input -> hidden1
    for (size_t i = 0; i < hidden_size; i++)
    {
        for (size_t j = 0; j < input_size; j++)
        {
            weights_0[i][j] += learning_rate * hidden1_error[i] * input[j];
        }
        bias_0[i] += learning_rate * hidden1_error[i];
    }

    // Update weights_1: hidden1 -> hidden2
    for (size_t i = 0; i < hidden_size; i++)
    {
        for (size_t j = 0; j < hidden_size; j++)
        {
            weights_1[i][j] += learning_rate * hidden2_error[i] * hidden1[j];
        }
        bias_1[i] += learning_rate * hidden2_error[i];
    }

    // Update weights_2: hidden2 -> output
    for (size_t i = 0; i < output_size; i++)
    {
        for (size_t j = 0; j < hidden_size; j++)
        {
            weights_2[i][j] += learning_rate * output_error[i] * hidden2[j];
        }
        bias_2[i] += learning_rate * output_error[i];
    }

    return max_index(output_size, output);
}

void stochastic_gradient_descent()
{
    float input[input_size];
    float label[output_size];
    float hidden[hidden_size];
    float hidden2[hidden_size];
    float output[output_size];

    int expected, guess;
    int correct = 0;

    for (size_t e = 0; e < epochs; e++)
    {
        correct = 0;

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

        learning_rate *= 0.99f; // Decay learning rate

        // Print progress
        fill_input(data, input_size, input, rand() % training_set_size);
        print_image(input);
        predict_debug(input, hidden, hidden2, output);

        guess = max_index(output_size, output);
        printf("Guess: %d\n", guess);
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

    init_layer(input_size, hidden_size, weights_0, bias_0);
    init_layer(hidden_size, hidden_size, weights_1, bias_1);
    init_layer(hidden_size, output_size, weights_2, bias_2);
}

void evaluate_model()
{
    // Test/evaluation data
    struct stat test_data_stat;
    unsigned char *test_data = NULL;
    unsigned char *test_labels = NULL;

    printf("\n=== Model Evaluation ===\n");

    // Load test data
    test_labels = read_labels("./data/t10k-labels.idx1-ubyte");
    test_data = open_dataset("./data/t10k-images.idx3-ubyte", &test_data_stat);

    if (test_labels == NULL || test_data == NULL)
    {
        printf("Error: Could not load test data\n");
        return;
    }

    float input[input_size];
    int correct = 0;
    int total_test_samples = 10000;

    // Confusion matrix for detailed analysis
    int confusion_matrix[output_size][output_size] = {0};

    printf("Evaluating on %d test samples...\n", total_test_samples);

    for (int i = 0; i < total_test_samples; i++)
    {
        fill_input(test_data, input_size, input, i);

        float hidden1[hidden_size];
        float hidden2[hidden_size];
        float output[output_size];

        predict(input, hidden1, hidden2, output);

        int predicted = max_index(output_size, output);
        int actual = test_labels[i];

        confusion_matrix[actual][predicted]++;

        if (predicted == actual)
        {
            correct++;
        }

        // Print progress every 1000 samples
        if ((i + 1) % 1000 == 0)
        {
            printf("Processed %d/%d samples\n", i + 1, total_test_samples);
        }
    }

    float accuracy = (float)correct / total_test_samples * 100.0f;
    printf("\n=== Results ===\n");
    printf("Test Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, total_test_samples);

    // Print per-class accuracy
    printf("\nPer-class accuracy:\n");
    for (int i = 0; i < output_size; i++)
    {
        int class_total = 0;
        int class_correct = confusion_matrix[i][i];

        for (int j = 0; j < output_size; j++)
        {
            class_total += confusion_matrix[i][j];
        }

        if (class_total > 0)
        {
            float class_accuracy = (float)class_correct / class_total * 100.0f;
            printf("Digit %d: %.2f%% (%d/%d)\n", i, class_accuracy, class_correct, class_total);
        }
    }

    // Print confusion matrix
    printf("\nConfusion Matrix:\n");
    printf("Actual\\Pred");
    for (int j = 0; j < output_size; j++)
    {
        printf("%5d", j);
    }
    printf("\n");

    for (int i = 0; i < output_size; i++)
    {
        printf("    %d      ", i);
        for (int j = 0; j < output_size; j++)
        {
            printf("%5d", confusion_matrix[i][j]);
        }
        printf("\n");
    }

    // Cleanup test data
    munmap(test_data, test_data_stat.st_size);
    free(test_labels);
}

int run_model(int argc, char const *argv[])
{
    printf("=== Neural Network Training ===\n");
    init();
    stochastic_gradient_descent();

    printf("\n=== Training Complete ===\n");

    // Evaluate the trained model
    evaluate_model();

    munmap(data, data_stat.st_size);
    free(labels);

    return 0;
}
