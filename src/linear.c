#include <stdio.h>

// Linear regression with one variable
// w = weights, b = bias
float w = 25.0f;
float b = 0.001f;

// training set
#define m 15
float x[m] = {1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83};
float y[m] = {52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46};

// #define m 3
// float x[m] = {1.0, 2.0, 3.0};
// float y[m] = {2.0, 3.0, 4.0};

float forward(float x)
{
    return w * x + b;
}

void gradient_descent(float x[m], float y[m])
{
    // The number of times we will update the weights and bias is called epochs.
    // The more epochs, the more accurate the model will be.
    const int epochs = 20000;

    // The learning rate (often α or alpha) below is chosen by trial and error to make
    // the function converge faster.
    // In practice this value is usually between 0.01 and 0.1
    const float learning_rate = 0.1;

    for (size_t e = 0; e < epochs; e++)
    {
        float cost = 0.0f;

        float dw = 0.0f;
        float db = 0.0f;

        for (size_t i = 0; i < m; i++)
        {
            float output = forward(x[i]);
            float error = y[i] - output;

            cost += error * error;

            dw += error * x[i];
            db += error;
        }

        cost /= 2 * m;

        dw /= m;
        db /= m;

        w += learning_rate * dw;
        b += learning_rate * db;

        if (e % 100 == 0)
        {
            printf("Epoch %zu, Cost: %f\n", e, cost);
            printf("w: %f, b: %f\n", w, b);
        }
    }
}

int main(int argc, char const *argv[])
{
    gradient_descent(x, y);

    printf("w: %f\n", w);
    printf("b: %f\n", b);

    printf("Prediction: %f\n", forward(1.85f));
    return 0;
}
