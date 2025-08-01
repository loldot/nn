#include <sys/stat.h>

unsigned char* read_labels(const char* filename);
unsigned char* open_dataset(const char* filename, struct stat *data_stat);
void fill_input(__uint8_t* addr, int n, float input[n], int offset);