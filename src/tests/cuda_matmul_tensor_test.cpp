#include <iostream>
#include <cmath>
#include "../tensor/tensor.h"
#include "../core/cuda_utils.h"

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    // Test 1: simple 2D matmul on GPU (A * B)
    // A: 2x3, B: 3x4, C: 2x4
    Tensor a({2, 3}, DType::Float32, Device::CUDA);
    Tensor b({3, 4}, DType::Float32, Device::CUDA);

    float a_h[6] = {1,2,3,4,5,6};
    float b_h[12] = {1,2,3,4,5,6,7,8,9,10,11,12};

    cudaMemcpy(a.data(), a_h, 6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.data(), b_h, 12*sizeof(float), cudaMemcpyHostToDevice);

    Tensor c = a.matmul(b);
    float c_h[8];
    cudaMemcpy(c_h, c.data(), 8*sizeof(float), cudaMemcpyDeviceToHost);

    // A*B = [[38,44,50,56], [83,98,113,128]]
    float expected[8] = {38,44,50,56, 83,98,113,128};
    bool ok = true;
    for (int i=0;i<8;++i) {
        if (std::abs(c_h[i] - expected[i]) > 1e-5) {
            std::cout << "mismatch at " << i << ": got " << c_h[i] << " expected " << expected[i] << "\n";
            ok = false;
        }
    }

    // Test 2: transposed A (A^T is 3x2)
    Tensor a2({3, 2}, DType::Float32, Device::CUDA);
    float a2_h[6] = {1,2,3,4,5,6};
    cudaMemcpy(a2.data(), a2_h, 6*sizeof(float), cudaMemcpyHostToDevice);
    Tensor a2_t = a2.transpose_view(0, 1); // shape 2x3, same as a
    Tensor c2 = a2_t.matmul(b);
    float c2_h[8];
    cudaMemcpy(c2_h, c2.data(), 8*sizeof(float), cudaMemcpyDeviceToHost);
    // a2 is [[1,2],[3,4],[5,6]], a2_t = [[1,3,5],[2,4,6]]
    float expected2[8] = {61,70,79,88, 76,88,100,112};
    bool ok2 = true;
    for (int i=0;i<8;++i) {
        if (std::abs(c2_h[i] - expected2[i]) > 1e-5) {
            std::cout << "transpose mismatch at " << i << ": got " << c2_h[i] << " expected " << expected2[i] << "\n";
            ok2 = false;
        }
    }

    // Test 3: batched 3D matmul
    Tensor a3({2, 2, 3}, DType::Float32, Device::CUDA);
    Tensor b3({2, 3, 4}, DType::Float32, Device::CUDA);
    float a3_h[12], b3_h[24];
    for (int i=0;i<12;++i) a3_h[i] = (float)i+1;
    for (int i=0;i<24;++i) b3_h[i] = (float)i+1;
    cudaMemcpy(a3.data(), a3_h, 12*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b3.data(), b3_h, 24*sizeof(float), cudaMemcpyHostToDevice);
    Tensor c3 = a3.matmul(b3);
    float c3_h[16];
    cudaMemcpy(c3_h, c3.data(), 16*sizeof(float), cudaMemcpyDeviceToHost);

    // batch 0: A=[[1,2,3],[4,5,6]], B=[[1..4],[5..8],[9..12]]
    // C[1,1] = 1*5+2*9+3*13 = 5+18+39 = 62 (if B is [3,4])
    // Actually B rows are 0:1,2,3,4; 1:5,6,7,8; 2:9,10,11,12
    // A[1,:] = [4,5,6]; C[1,j] = 4*B[0,j] + 5*B[1,j] + 6*B[2,j]
    // j=0: 4*1 + 5*5 + 6*9 = 4+25+54 = 83
    // j=1: 4*2 + 5*6 + 6*10 = 8+30+60 = 98
    float expected3[16] = {
        38,44,50,56, 83,98,113,128,
        // batch 1 values from a3_h[6..11] = 7,8,9,10,11,12 and b3_h[12..23] = 13..24
        // A[0,:] = [7,8,9]; A[1,:] = [10,11,12]
        // B row 0: 13,14,15,16; row1:17,18,19,20; row2:21,22,23,24
        // batch 1, A[1] dot B rows:
        // j=0: 10*13+11*17+12*21 = 130+187+252 = 569
        // j=1: 10*14+11*18+12*22 = 140+198+264 = 602
        // j=2: 10*15+11*19+12*23 = 150+209+276 = 635
        // j=3: 10*16+11*20+12*24 = 160+220+288 = 668
    };
    expected3[8]=7*13+8*17+9*21;  // 91+136+189 = 416
    expected3[9]=7*14+8*18+9*22;  // 98+144+198 = 440
    expected3[10]=7*15+8*19+9*23; // 105+152+207 = 464
    expected3[11]=7*16+8*20+9*24; // 112+160+216 = 488
    expected3[12]=569;
    expected3[13]=602;
    expected3[14]=635;
    expected3[15]=668;

    bool ok3 = true;
    for (int i=0;i<16;++i) {
        if (std::abs(c3_h[i] - expected3[i]) > 1e-4) {
            std::cout << "batch mismatch at " << i << ": got " << c3_h[i] << " expected " << expected3[i] << "\n";
            ok3 = false;
        }
    }

    if (ok && ok2 && ok3) {
        std::cout << "All CUDA matmul checks passed.\n";
        return 0;
    }
    return 1;
}
