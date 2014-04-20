#include <stdio.h>
#include <stdlib.h>

int* parse_data(FILE *f, int n) {
	int* data = (int*) malloc(sizeof(int)*n);
	int i;
	for (i = 0; i < n; i++) {
		data[i] = fgetc(f);
	}
	return data;
}