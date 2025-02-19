#define input_size 784
#define hidden_size 16
#define output_size 10

#define epochs 30
#define traing_set_size 60000
#define learning_rate 0.01

const int max_index(const double* tensor, int size){
    int max = 0;
    for (size_t i = 0; i < size; i++)
    {
        max = (tensor[i] > tensor[max]) ? i : max;
    }
    return max;
}