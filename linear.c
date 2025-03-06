#include <stdio.h>

// Linear regression with one variable
// w = weights, b = bias
float w = 0.001f;
float b = 0.001f;

// training set
// #define m 15
// float x[m] = {1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83};
// float y[m] = {52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46};

#define m 3
float x[m] = {1.0, 2.0, 3.0};
float y[m] = {2.0, 3.0, 4.0};

float forward(float x)
{
    return w * x + b;
}

int main(int argc, char const *argv[])
{
    const float learning_rate = 0.01f;
    const int epochs = 10000;

    for (size_t i = 0; i < epochs; i++)
    {
        float cost = 0.0f;

        float dw = 0.0f;
        float db = 0.0f;

        for (size_t j = 0; j < m; j++)
        {
            float output = forward(x[j]);
            float error = y[j] - output;

            cost += error * error;

            dw += error * x[j];
            db += error;
        }

        cost /= 2 * m;
        printf("Cost: %f\n", cost);

        dw /= m;
        db /= m;
        
        w += learning_rate * dw;
        b += learning_rate * db;
    }

    printf("w: %f\n", w);
    printf("b: %f\n", b);

    printf("Prediction: %f\n", forward(4.0f));
    return 0;
}
