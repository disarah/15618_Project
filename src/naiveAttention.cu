#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>

#include "naiveAttention.h"

__global__ void test_kernel(void) {
}

void wrapper(void) {
    std::cout << "here!\n";
	test_kernel <<<1, 1 >>> ();
}
