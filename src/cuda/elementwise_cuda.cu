#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#endif
#include <cuda_runtime.h>
#include <cstdint>
#include "../core/dtype_utils.h"
#include <mma.h>
using namespace nvcuda;

// Elementwise add kernel
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Elementwise sub kernel
template<typename T>
__global__ void sub_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

// Elementwise mul kernel
template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

// Elementwise div kernel
template<typename T>
__global__ void div_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

// ReLU activation kernel
template<typename T>
__global__ void relu_kernel(const T* in, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] > 0 ? in[idx] : T{};
    }
}

// Naive matrix multiplication kernel
// C = A * B where A is (M x K), B is (K x N), C is (M x N)
template<typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum{};
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template<typename T>
__global__ void matmul_kernel_shared_memory(const T* A, const T* B, T* C, int M, int K, int N) {
    const int TILE = 16;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ T tileA[TILE][TILE];
    __shared__ T tileB[TILE][TILE];

    T sum = T{};

    for (int tile = 0; tile < K; tile += TILE) {
        if (row < M && tile + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tile + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = T{};

        if (tile + threadIdx.y < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = T{};

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

template<typename T>
__global__ void matmul_kernel_register_blcoking(const T* A, const T* B, T* C, int M, int K, int N) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;
    constexpr int THREAD_TILE = 8;

    int rowCorner = blockIdx.y * BLOCK_M + threadIdx.y * THREAD_TILE;
    int colCorner = blockIdx.x * BLOCK_N + threadIdx.x * THREAD_TILE;


    __shared__ T tileA[BLOCK_M][BK];
    __shared__ T tileB[BK][BLOCK_N];

    T accum[THREAD_TILE][THREAD_TILE] = {};

    for (int tile = 0; tile < K; tile += BK) {
        for (int i = 0; i < THREAD_TILE; ++i){
            int sharedRow = threadIdx.y * THREAD_TILE + i; // 0..127
            int sharedCol = threadIdx.x;    

            int globalRow = blockIdx.y * BLOCK_M + sharedRow;
            int globalCol = tile + sharedCol;
           
            if (globalRow < M && globalCol < K)
                tileA[sharedRow][sharedCol] =
                    A[globalRow * K + globalCol];
            else
                tileA[sharedRow][sharedCol] = T{};
        }
        for (int i = 0; i < THREAD_TILE; ++i){
            int sharedRow = threadIdx.y;
            int sharedCol = threadIdx.x * THREAD_TILE + i;

            int globalRow = tile + sharedRow;
            int globalCol = blockIdx.x * BLOCK_N + sharedCol;

            if (globalRow < K && globalCol < N)
                tileB[sharedRow][sharedCol] =
                    B[globalRow * N + globalCol];
            else
                tileB[sharedRow][sharedCol] = T{};
        }

        __syncthreads();

        T A_register[THREAD_TILE];
        T B_register[THREAD_TILE];

        for (int k = 0; k < BK; ++k) {
            for(int i=0; i < THREAD_TILE; ++i){
                A_register[i] = tileA[threadIdx.y * THREAD_TILE + i][k];
                B_register[i] = tileB[k][threadIdx.x * THREAD_TILE + i];
            }
            
            for(int r=0; r<THREAD_TILE; ++r){
                for(int c=0; c<THREAD_TILE; ++c){
                   accum[r][c] += A_register[r] * B_register[c]; 
                }
            }
        }

        __syncthreads();
    }

    for(int r=0; r<THREAD_TILE; ++r){
        for(int c=0; c<THREAD_TILE; ++c){
            int row = rowCorner+r;
            int col = colCorner+c;
            if (row < M && col < N){
                C[row * N + col] = accum[r][c];
            }
        }
    }
}

template<typename T>
__global__ void matmul_kernel_vectorized_input (const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K, int N) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;
    constexpr int THREAD_TILE = 8;

    constexpr int uint4Size = 16; //an uint4 can hold 16 bytes of data
    constexpr int vectorSize = uint4Size / sizeof(T); //how many element a uint4 number represent
    static_assert(
        THREAD_TILE % vectorSize == 0
    );
    constexpr int loopNum = THREAD_TILE / vectorSize; //how many times each thread iterate loading a uint4Size every time.
    constexpr int SMEM_PAD_A = 0; // keep A rows 16-byte aligned for uint4
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ T tileA[BLOCK_M][BK + SMEM_PAD_A];
    __shared__ T tileB[BK][BLOCK_N];

    T accum[THREAD_TILE][THREAD_TILE] = {};

    T A_register[THREAD_TILE];
    T B_register[THREAD_TILE];

    for (int tile = 0; tile < K; tile += BK) {
        for (int i=0; i<loopNum; ++i){
            int vectorLoadId = tid * loopNum + i;
            int linearId = vectorLoadId * vectorSize;

            int sharedRow = linearId / BK;
            int sharedCol = linearId % BK;

            int globalRow = blockIdx.y * BLOCK_M + sharedRow;
            int globalCol = tile + sharedCol;

            if (globalCol + vectorSize <= K && globalRow < M){
                *reinterpret_cast<uint4*>(&tileA[sharedRow][sharedCol]) =
                            *reinterpret_cast<const uint4*>(&A[globalRow * K + globalCol]);      
            }
            else{
                for(int j=0; j<vectorSize; ++j){
                    if (globalRow < M && globalCol+j < K)
                        tileA[sharedRow][sharedCol+j] =
                            A[globalRow * K + globalCol + j];
                    else
                        tileA[sharedRow][sharedCol+j] = T{}; 
                }
            }          
        }

        for (int i=0; i<loopNum; ++i){
            int vectorLoadId = tid * loopNum + i;
            int linearId = vectorLoadId * vectorSize;

            int sharedRow = linearId / BLOCK_N;
            int sharedCol = linearId % BLOCK_N;


            int globalRow = tile + sharedRow;
            int globalCol = blockIdx.x * BLOCK_N + sharedCol;

            if (globalCol + vectorSize <= N && globalRow < K){
                *reinterpret_cast<uint4*>(&tileB[sharedRow][sharedCol]) =
                            *reinterpret_cast<const uint4*>(&B[globalRow * N + globalCol]);      
            }
            else{
                for(int j=0; j<vectorSize; ++j){
                    if (globalRow < K && globalCol+j < N)
                        tileB[sharedRow][sharedCol+j] =
                            B[globalRow * N + globalCol + j];
                    else
                        tileB[sharedRow][sharedCol+j] = T{}; 
                }
            }          
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            
            #pragma unroll
            for(int i=0; i < THREAD_TILE; ++i){
                A_register[i] = tileA[threadIdx.y * THREAD_TILE + i][k];
                B_register[i] = tileB[k][threadIdx.x * THREAD_TILE + i];
            }

            #pragma unroll
            for(int r=0; r<THREAD_TILE; ++r){
                for(int c=0; c<THREAD_TILE; ++c){
                   accum[r][c] += A_register[r] * B_register[c]; 
                }
            }
        }

        __syncthreads();
    }

    int rowCorner = blockIdx.y * BLOCK_M + threadIdx.y * THREAD_TILE;
    int colCorner = blockIdx.x * BLOCK_N + threadIdx.x * THREAD_TILE;

    for(int r=0; r<THREAD_TILE; ++r){
        for(int c=0; c<THREAD_TILE; ++c){
            int row = rowCorner+r;
            int col = colCorner+c;
            if (row < M && col < N){
                C[row * N + col] = accum[r][c];
            }
        }
    }
}

template<typename T>
__global__ void matmul_kernel_warp_tiling(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M,
    int K,
    int N
)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;


    // Warp configuration
    constexpr int WARP_ROWS = 4;
    constexpr int WARP_COLS = 2;

    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS; //32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS; //64


    // Lane grouping
    constexpr int GROUPS = 4;
    constexpr int LANES_PER_GROUP = 8;


    // Thread output tile
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;


    constexpr int UINT4_BYTES = 16;
    constexpr int VECTOR_SIZE = UINT4_BYTES / sizeof(T);
    constexpr int SMEM_PAD_A = 0; // keep A rows 16-byte aligned for uint4

    static_assert(
        THREAD_TILE_N % VECTOR_SIZE == 0
    );

    //the linear id of a thread
    //we define a block as a linear array of thread so only the x cordinate matter
    int tid = threadIdx.x;

    
    int warpId = tid / 32;
    int laneId = tid % 32;


    int warpRow = warpId / WARP_COLS;
    int warpCol = warpId % WARP_COLS;


    int groupId = laneId / LANES_PER_GROUP;
    int laneInGroup = laneId % LANES_PER_GROUP;



    /*
        Thread owns:

        warp tile
            +
        group row offset
            +
        lane column offset
    */


    int threadRow =
        warpRow * WARP_TILE_M
        +
        groupId * THREAD_TILE_M;


    int threadCol =
        warpCol * WARP_TILE_N
        +
        laneInGroup * THREAD_TILE_N;



    __shared__ T tileA[BLOCK_M][BK + SMEM_PAD_A];
    __shared__ T tileB[BK][BLOCK_N];



    T accum[THREAD_TILE_M][THREAD_TILE_N] = {};


    T A_reg[THREAD_TILE_M];
    T B_reg[THREAD_TILE_N];



    for(int tile = 0; tile < K; tile += BK)
    {

        /*
            Load A

            128×16 = 2048 elements

            256 threads
            each thread loads 8 values
        */


        int loadId = tid * VECTOR_SIZE;

        for(int i = loadId; i < BLOCK_M * BK; i += blockDim.x * blockDim.y * VECTOR_SIZE) {

            int row = i / BK;
            int col = i % BK;


            int globalRow =
                blockIdx.y * BLOCK_M + row;

            int globalCol =
                tile + col;



            if(globalRow < M &&
               globalCol + VECTOR_SIZE <= K)
            {

                *reinterpret_cast<uint4*>(
                    &tileA[row][col]
                )
                =
                *reinterpret_cast<const uint4*>(
                    &A[globalRow*K + globalCol]
                );

            }
            else
            {

                #pragma unroll
                for(int j=0;j<VECTOR_SIZE;j++)
                {
                    if(globalRow < M &&
                       globalCol+j < K)
                    {
                        tileA[row][col+j]
                            =
                        A[globalRow*K+globalCol+j];
                    }
                    else
                    {
                        tileA[row][col+j]=T{};
                    }
                }
            }
        }




        /*
            Load B

            16×128 = 2048 elements
        */


        for(int i = loadId;
            i < BK*BLOCK_N;
            i += blockDim.x * blockDim.y * VECTOR_SIZE)
        {

            int row = i / BLOCK_N;
            int col = i % BLOCK_N;


            int globalRow =
                tile + row;

            int globalCol =
                blockIdx.x*BLOCK_N + col;



            if(globalRow < K &&
               globalCol + VECTOR_SIZE <= N)
            {

                *reinterpret_cast<uint4*>(
                    &tileB[row][col]
                )
                =
                *reinterpret_cast<const uint4*>(
                    &B[globalRow*N+globalCol]
                );

            }
            else
            {

                #pragma unroll
                for(int j=0;j<VECTOR_SIZE;j++)
                {
                    if(globalRow<K &&
                       globalCol+j<N)
                    {
                        tileB[row][col+j]
                            =
                        B[globalRow*N+globalCol+j];
                    }
                    else
                    {
                        tileB[row][col+j]=T{};
                    }
                }
            }

        }



        __syncthreads();



        /*
            Compute

            Each thread:
                8×8 output

        */


        #pragma unroll
        for(int k=0;k<BK;k++)
        {


            #pragma unroll
            for(int i=0;i<THREAD_TILE_M;i++)
            {
                A_reg[i] =
                    tileA[
                        threadRow+i
                    ][k];
            }


            #pragma unroll
            for(int i=0;i<THREAD_TILE_N;i++)
            {
                B_reg[i] =
                    tileB[
                        k
                    ][
                        threadCol+i
                    ];
            }



            #pragma unroll
            for(int r=0;r<THREAD_TILE_M;r++)
            {
                #pragma unroll
                for(int c=0;c<THREAD_TILE_N;c++)
                {
                    accum[r][c]
                    +=
                    A_reg[r]
                    *
                    B_reg[c];
                }
            }

        }


        __syncthreads();

    }




    /*
        Store output
    */


    int rowStart =
        blockIdx.y*BLOCK_M
        +
        threadRow;


    int colStart = blockIdx.x * BLOCK_N + threadCol;



    #pragma unroll
    for(int r=0;r<THREAD_TILE_M;r++)
    {

        #pragma unroll
        for(int c=0;c<THREAD_TILE_N;c++)
        {

            int row=rowStart+r;
            int col=colStart+c;


            if(row<M && col<N)
            {
                C[row*N+col]
                    =
                accum[r][c];
            }

        }

    }
}

template<typename T>
__global__ void matmul_kernel_double_buffered(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int K, int N
)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;
    
    constexpr int WARP_ROWS = 4;
    constexpr int WARP_COLS = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS;  // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS;  // 64
    
    constexpr int GROUPS = 4;
    constexpr int LANES_PER_GROUP = 8;
    
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;
    
    constexpr int UINT4_BYTES = 16;
    constexpr int VECTOR_SIZE = UINT4_BYTES / sizeof(T);
    constexpr int SMEM_PAD_A = 0; // keep A rows 16-byte aligned for uint4
    
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    int warpRow = warpId / WARP_COLS;
    int warpCol = warpId % WARP_COLS;
    
    int groupId = laneId / LANES_PER_GROUP;
    int laneInGroup = laneId % LANES_PER_GROUP;
    
    int threadRow = warpRow * WARP_TILE_M + groupId * THREAD_TILE_M;
    int threadCol = warpCol * WARP_TILE_N + laneInGroup * THREAD_TILE_N;
    
    // ========== DOUBLE BUFFERING: TWO SHARED MEMORY BUFFERS ==========
    __shared__ T tileA[2][BLOCK_M][BK + SMEM_PAD_A];
    __shared__ T tileB[2][BK][BLOCK_N];
    // ===================================================================
    
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {};
    T A_reg[THREAD_TILE_M];
    T B_reg[THREAD_TILE_N];
    
    // ========== PROLOGUE: LOAD FIRST TILE ==========
    {
        int bufferIdx = 0;
        int tile = 0;
        int loadId = tid * VECTOR_SIZE;
        
        // Load A into buffer 0
        for(int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = blockIdx.y * BLOCK_M + row;
            int globalCol = tile + col;
            
            if(globalRow < M && globalCol + VECTOR_SIZE <= K) {
                *reinterpret_cast<uint4*>(&tileA[bufferIdx][row][col])
                    = *reinterpret_cast<const uint4*>(&A[globalRow*K + globalCol]);
            } else {
                #pragma unroll
                for(int j = 0; j < VECTOR_SIZE; j++) {
                    tileA[bufferIdx][row][col+j] = 
                        (globalRow < M && globalCol+j < K) 
                            ? A[globalRow*K + globalCol+j]
                            : T{};
                }
            }
        }
        
        // Load B into buffer 0
        for(int i = loadId; i < BK*BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRow = tile + row;
            int globalCol = blockIdx.x*BLOCK_N + col;
            
            if(globalRow < K && globalCol + VECTOR_SIZE <= N) {
                *reinterpret_cast<uint4*>(&tileB[bufferIdx][row][col])
                    = *reinterpret_cast<const uint4*>(&B[globalRow*N + globalCol]);
            } else {
                #pragma unroll
                for(int j = 0; j < VECTOR_SIZE; j++) {
                    tileB[bufferIdx][row][col+j] = 
                        (globalRow < K && globalCol+j < N) 
                            ? B[globalRow*N + globalCol+j]
                            : T{};
                }
            }
        }
        
        __syncthreads();
    }
    
    // ========== MAIN LOOP: LOAD & COMPUTE IN PARALLEL ==========
    for(int tile = 0; tile < K; tile += BK)
    {
        int bufferIdx = (tile / BK) % 2;
        int nextTile = tile + BK;
        int nextBufferIdx = 1 - bufferIdx;
        
        // ===== LOAD NEXT TILE (into nextBufferIdx) =====
        if(nextTile < K)  // Only if there's a next tile
        {
            int loadId = tid * VECTOR_SIZE;
            
            // Load A into next buffer
            for(int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
                int row = i / BK;
                int col = i % BK;
                int globalRow = blockIdx.y * BLOCK_M + row;
                int globalCol = nextTile + col;
                
                if(globalRow < M && globalCol + VECTOR_SIZE <= K) {
                    *reinterpret_cast<uint4*>(&tileA[nextBufferIdx][row][col])
                        = *reinterpret_cast<const uint4*>(&A[globalRow*K + globalCol]);
                } else {
                    #pragma unroll
                    for(int j = 0; j < VECTOR_SIZE; j++) {
                        tileA[nextBufferIdx][row][col+j] = 
                            (globalRow < M && globalCol+j < K) 
                                ? A[globalRow*K + globalCol+j]
                                : T{};
                    }
                }
            }
            
            // Load B into next buffer
            for(int i = loadId; i < BK*BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                int globalRow = nextTile + row;
                int globalCol = blockIdx.x*BLOCK_N + col;
                
                if(globalRow < K && globalCol + VECTOR_SIZE <= N) {
                    *reinterpret_cast<uint4*>(&tileB[nextBufferIdx][row][col])
                        = *reinterpret_cast<const uint4*>(&B[globalRow*N + globalCol]);
                } else {
                    #pragma unroll
                    for(int j = 0; j < VECTOR_SIZE; j++) {
                        tileB[nextBufferIdx][row][col+j] = 
                            (globalRow < K && globalCol+j < N) 
                                ? B[globalRow*N + globalCol+j]
                                : T{};
                    }
                }
            }
        }
        
        // ===== COMPUTE FROM CURRENT BUFFER (bufferIdx) =====
        #pragma unroll
        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; i++) {
                A_reg[i] = tileA[bufferIdx][threadRow+i][k];
            }
            
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_N; i++) {
                B_reg[i] = tileB[bufferIdx][k][threadCol+i];
            }
            
            #pragma unroll
            for(int r = 0; r < THREAD_TILE_M; r++) {
                #pragma unroll
                for(int c = 0; c < THREAD_TILE_N; c++) {
                    accum[r][c] += A_reg[r] * B_reg[c];
                }
            }
        }
        
        __syncthreads();  // Ensure load is done before next iteration
    }
    
    // ========== STORE OUTPUT ==========
    int rowStart = blockIdx.y*BLOCK_M + threadRow;
    int colStart = blockIdx.x*BLOCK_N + threadCol;
    
    #pragma unroll
    for(int r = 0; r < THREAD_TILE_M; r++) {
        #pragma unroll
        for(int c = 0; c < THREAD_TILE_N; c++) {
            int row = rowStart + r;
            int col = colStart + c;
            if(row < M && col < N) {
                C[row*N + col] = accum[r][c];
            }
        }
    }
}

#include <cuda_runtime.h>

// ========== CP.ASYNC PTX PRIMITIVES ==========
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* global_ptr, bool valid, int remaining_bytes) {
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    int src_size = valid ? max(0, min(16, remaining_bytes)) : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(global_ptr), "r"(src_size)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
// ============================================

template<typename T>
__global__ void matmul_kernel_double_buffered_cpasync(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int K, int N
)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;
    
    constexpr int WARP_ROWS = 4;
    constexpr int WARP_COLS = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS;  // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS;  // 64
    
    constexpr int GROUPS = 4;
    constexpr int LANES_PER_GROUP = 8;
    
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;
    
    constexpr int UINT4_BYTES = 16;
    constexpr int VECTOR_SIZE = UINT4_BYTES / sizeof(T);
    
    // IMPORTANT: Padding must align rows to 16 bytes for cp.async.
    // For float (4 bytes), VECTOR_SIZE = 4 elements (16 bytes).
    constexpr int SMEM_PAD_A = VECTOR_SIZE; 

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    int warpRow = warpId / WARP_COLS;
    int warpCol = warpId % WARP_COLS;
    
    int groupId = laneId / LANES_PER_GROUP;
    int laneInGroup = laneId % LANES_PER_GROUP;
    
    int threadRow = warpRow * WARP_TILE_M + groupId * THREAD_TILE_M;
    int threadCol = warpCol * WARP_TILE_N + laneInGroup * THREAD_TILE_N;
    
    // ========== DOUBLE BUFFERING: TWO SHARED MEMORY BUFFERS ==========
    __shared__ T tileA[2][BLOCK_M][BK + SMEM_PAD_A];
    __shared__ T tileB[2][BK][BLOCK_N];
    // ===================================================================
    
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {};
    T A_reg[THREAD_TILE_M];
    T B_reg[THREAD_TILE_N];
    
    int loadId = tid * VECTOR_SIZE;

    // Helper device lambdas for async loading
    auto load_A = [&](int buf_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = blockIdx.y * BLOCK_M + row;
            int globalCol = current_k_tile + col;
            
            bool valid = (globalRow < M) && (globalCol < K);
            int remaining_bytes = (K - globalCol) * static_cast<int>(sizeof(T));
            
            cp_async_16(
                &tileA[buf_idx][row][col],
                &A[globalRow * K + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    auto load_B = [&](int buf_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BK * BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRow = current_k_tile + row;
            int globalCol = blockIdx.x * BLOCK_N + col;
            
            bool valid = (globalRow < K) && (globalCol < N);
            int remaining_bytes = (N - globalCol) * static_cast<int>(sizeof(T));
            
            cp_async_16(
                &tileB[buf_idx][row][col],
                &B[globalRow * N + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    // ========== PROLOGUE: PREFETCH TILE 0 ASYNCHRONOUSLY ==========
    load_A(0, 0);
    load_B(0, 0);
    cp_async_commit();
    cp_async_wait<0>(); // Wait for Tile 0 transfer to finish
    __syncthreads();
    
    // ========== MAIN LOOP: OVERLAP LOAD & COMPUTE ==========
    for(int tile = 0; tile < K; tile += BK)
    {
        int bufferIdx = (tile / BK) % 2;
        int nextTile = tile + BK;
        int nextBufferIdx = 1 - bufferIdx;
        
        // 1. ISSUE ASYNC LOAD FOR NEXT TILE (into nextBufferIdx)
        if(nextTile < K) 
        {
            load_A(nextBufferIdx, nextTile);
            load_B(nextBufferIdx, nextTile);
            cp_async_commit(); // Group transfers for nextTile
        }
        
        // 2. COMPUTE CURRENT TILE FROM SHARED MEMORY (bufferIdx)
        #pragma unroll
        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; i++) {
                A_reg[i] = tileA[bufferIdx][threadRow + i][k];
            }
            
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_N; i++) {
                B_reg[i] = tileB[bufferIdx][k][threadCol + i];
            }
            
            #pragma unroll
            for(int r = 0; r < THREAD_TILE_M; r++) {
                #pragma unroll
                for(int c = 0; c < THREAD_TILE_N; c++) {
                    accum[r][c] += A_reg[r] * B_reg[c];
                }
            }
        }
        
        // 3. SYNCHRONIZE NEXT TILE ASYNC LOAD BEFORE SWAPPING
        if(nextTile < K) 
        {
            cp_async_wait<0>(); // Block until next tile load completes
            __syncthreads();    // Ensure all threads in the block are ready
        }
    }
    
    // ========== STORE OUTPUT ==========
    int rowStart = blockIdx.y * BLOCK_M + threadRow;
    int colStart = blockIdx.x * BLOCK_N + threadCol;
    
    #pragma unroll
    for(int r = 0; r < THREAD_TILE_M; r++) {
        #pragma unroll
        for(int c = 0; c < THREAD_TILE_N; c++) {
            int row = rowStart + r;
            int col = colStart + c;
            if(row < M && col < N) {
                C[row * N + col] = accum[r][c];
            }
        }
    }
}

// ========== SWIZZLE HELPER FUNCTION ==========
// Maps 16-byte vector chunks into swizzled physical bank locations.
// Eliminates 4-way bank conflicts across thread groups while preserving 16-byte alignment.
template<typename T>
__device__ __forceinline__ int swizzle_col_A(int row, int col) {
    constexpr int ELEMS_PER_16B = 16 / sizeof(T);
    int vec_col = col / ELEMS_PER_16B;        // 16-byte vector chunk index
    int off_col = col % ELEMS_PER_16B;        // Element index within the 16-byte vector
    int num_vec = 16 / ELEMS_PER_16B;         // number of 16-byte chunks in BK=16
    int swizzle_pattern = vec_col ^ ((row / 8) % num_vec);
    return swizzle_pattern * ELEMS_PER_16B + off_col;
}
// ============================================

template<typename T>
__global__ void matmul_kernel_double_buffered_swizzled(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int K, int N
)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;
    
    constexpr int WARP_ROWS = 4;
    constexpr int WARP_COLS = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS;  // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS;  // 64
    
    constexpr int GROUPS = 4;
    constexpr int LANES_PER_GROUP = 8;
    
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;
    
    constexpr int UINT4_BYTES = 16;
    constexpr int VECTOR_SIZE = UINT4_BYTES / sizeof(T);
    
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    int warpRow = warpId / WARP_COLS;
    int warpCol = warpId % WARP_COLS;
    
    int groupId = laneId / LANES_PER_GROUP;
    int laneInGroup = laneId % LANES_PER_GROUP;
    
    int threadRow = warpRow * WARP_TILE_M + groupId * THREAD_TILE_M;
    int threadCol = warpCol * WARP_TILE_N + laneInGroup * THREAD_TILE_N;
    
    // ========== DOUBLE BUFFERING: TWO SHARED MEMORY BUFFERS ==========
    __shared__ T tileA[2][BLOCK_M][BK];
    __shared__ T tileB[2][BK][BLOCK_N];
    // ===================================================================
    
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {};
    T A_reg[THREAD_TILE_M];
    T B_reg[THREAD_TILE_N];
    
    int loadId = tid * VECTOR_SIZE;

    // Helper device lambdas for async loading
    auto load_A = [&](int buf_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = blockIdx.y * BLOCK_M + row;
            int globalCol = current_k_tile + col;
            
            bool valid = (globalRow < M) && (globalCol < K);
            int remaining_bytes = (K - globalCol) * static_cast<int>(sizeof(T));
            
            // SWIZZLED STORE: Write into swizzled shared memory address
            int swizzled_col = swizzle_col_A<T>(row, col);
            
            cp_async_16(
                &tileA[buf_idx][row][swizzled_col],
                &A[globalRow * K + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    auto load_B = [&](int buf_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BK * BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRow = current_k_tile + row;
            int globalCol = blockIdx.x * BLOCK_N + col;
            
            bool valid = (globalRow < K) && (globalCol < N);
            int remaining_bytes = (N - globalCol) * static_cast<int>(sizeof(T));
            
            cp_async_16(
                &tileB[buf_idx][row][col],
                &B[globalRow * N + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    // ========== PROLOGUE: PREFETCH TILE 0 ASYNCHRONOUSLY ==========
    load_A(0, 0);
    load_B(0, 0);
    cp_async_commit();
    cp_async_wait<0>();   // wait for all pending async ops
    __syncthreads();
    
    // ========== MAIN LOOP: OVERLAP LOAD & COMPUTE ==========
    for(int tile = 0; tile < K; tile += BK)
    {
        int bufferIdx = (tile / BK) % 2;
        int nextTile = tile + BK;
        int nextBufferIdx = 1 - bufferIdx;
        
        // 1. ISSUE ASYNC LOAD FOR NEXT TILE
        if(nextTile < K) 
        {
            load_A(nextBufferIdx, nextTile);
            load_B(nextBufferIdx, nextTile);
            cp_async_commit();
        }
        
        // 2. COMPUTE CURRENT TILE FROM SHARED MEMORY
        #pragma unroll
        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; i++) {
                // SWIZZLED READ: Match the swizzled layout used during load
                int swizzled_k = swizzle_col_A<T>(threadRow + i, k);
                A_reg[i] = tileA[bufferIdx][threadRow + i][swizzled_k];
            }
            
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_N; i++) {
                B_reg[i] = tileB[bufferIdx][k][threadCol + i];
            }
            
            #pragma unroll
            for(int r = 0; r < THREAD_TILE_M; r++) {
                #pragma unroll
                for(int c = 0; c < THREAD_TILE_N; c++) {
                    accum[r][c] += A_reg[r] * B_reg[c];
                }
            }
        }
        
        // 3. SYNCHRONIZE NEXT TILE ASYNC LOAD BEFORE SWAPPING
        if(nextTile < K) 
        {
            cp_async_wait<0>();
            __syncthreads();
        }
    }
    
    // ========== STORE OUTPUT ==========
    int rowStart = blockIdx.y * BLOCK_M + threadRow;
    int colStart = blockIdx.x * BLOCK_N + threadCol;
    
    #pragma unroll
    for(int r = 0; r < THREAD_TILE_M; r++) {
        #pragma unroll
        for(int c = 0; c < THREAD_TILE_N; c++) {
            int row = rowStart + r;
            int col = colStart + c;
            if(row < M && col < N) {
                C[row * N + col] = accum[r][c];
            }
        }
    }
}

template<typename T>
__global__ void matmul_kernel_vector_storage (
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int K, int N
)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;
    
    constexpr int WARP_ROWS = 4;
    constexpr int WARP_COLS = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS;  // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS;  // 64
    
    constexpr int GROUPS = 4;
    constexpr int LANES_PER_GROUP = 8;
    
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;
    
    constexpr int UINT4_BYTES = 16;
    constexpr int VECTOR_SIZE = UINT4_BYTES / sizeof(T);
    
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    int warpRow = warpId / WARP_COLS;
    int warpCol = warpId % WARP_COLS;
    
    int groupId = laneId / LANES_PER_GROUP;
    int laneInGroup = laneId % LANES_PER_GROUP;
    
    int threadRow = warpRow * WARP_TILE_M + groupId * THREAD_TILE_M;
    int threadCol = warpCol * WARP_TILE_N + laneInGroup * THREAD_TILE_N;
    
    // ========== DOUBLE BUFFERING: TWO SHARED MEMORY BUFFERS ==========
    __shared__ T tileA[2][BLOCK_M][BK];
    __shared__ T tileB[2][BK][BLOCK_N];
    // ===================================================================
    
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {};
    T A_reg[THREAD_TILE_M];
    T B_reg[THREAD_TILE_N];
    
    int loadId = tid * VECTOR_SIZE;

    // Helper device lambdas for async loading
    auto load_A = [&](int buf_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = blockIdx.y * BLOCK_M + row;
            int globalCol = current_k_tile + col;
            
            bool valid = (globalRow < M) && (globalCol < K);
            int remaining_bytes = (K - globalCol) * static_cast<int>(sizeof(T));
            
            // SWIZZLED STORE: Write into swizzled shared memory address
            int swizzled_col = swizzle_col_A<T>(row, col);
            
            cp_async_16(
                &tileA[buf_idx][row][swizzled_col],
                &A[globalRow * K + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    auto load_B = [&](int buf_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BK * BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRow = current_k_tile + row;
            int globalCol = blockIdx.x * BLOCK_N + col;
            
            bool valid = (globalRow < K) && (globalCol < N);
            int remaining_bytes = (N - globalCol) * static_cast<int>(sizeof(T));
            
            cp_async_16(
                &tileB[buf_idx][row][col],
                &B[globalRow * N + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    // ========== PROLOGUE: PREFETCH TILE 0 ASYNCHRONOUSLY ==========
    load_A(0, 0);
    load_B(0, 0);
    cp_async_commit();
    cp_async_wait<0>(); 
    __syncthreads();
    
    // ========== MAIN LOOP: OVERLAP LOAD & COMPUTE ==========
    for(int tile = 0; tile < K; tile += BK)
    {
        int bufferIdx = (tile / BK) % 2;
        int nextTile = tile + BK;
        int nextBufferIdx = 1 - bufferIdx;
        
        // 1. ISSUE ASYNC LOAD FOR NEXT TILE
        if(nextTile < K) 
        {
            load_A(nextBufferIdx, nextTile);
            load_B(nextBufferIdx, nextTile);
            cp_async_commit();
        }
        
        // 2. COMPUTE CURRENT TILE FROM SHARED MEMORY
        #pragma unroll
        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; i++) {
                // SWIZZLED READ: Match the swizzled layout used during load
                int swizzled_k = swizzle_col_A<T>(threadRow + i, k);
                A_reg[i] = tileA[bufferIdx][threadRow + i][swizzled_k];
            }
            
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_N; i++) {
                B_reg[i] = tileB[bufferIdx][k][threadCol + i];
            }
            
            #pragma unroll
            for(int r = 0; r < THREAD_TILE_M; r++) {
                #pragma unroll
                for(int c = 0; c < THREAD_TILE_N; c++) {
                    accum[r][c] += A_reg[r] * B_reg[c];
                }
            }
        }
        
        // 3. SYNCHRONIZE NEXT TILE ASYNC LOAD BEFORE SWAPPING
        if(nextTile < K) 
        {
            cp_async_wait<0>();
            __syncthreads();
        }
    }
        
    // =========================================================================
    //        1D-BLOCK WARP-TILED VECTORIZED EPILOGUE (STORES TO C)
    // =========================================================================

    // 5. Calculate Global Matrix C Starting Coordinates
    int rowStart = blockIdx.y * BLOCK_M + threadRow;
    int colStart = blockIdx.x * BLOCK_N + threadCol;

    // Number of 16-byte vector chunks per thread row
    // For float (4 bytes): VECTOR_SIZE = 4 -> VECTORS_PER_ROW = 8 / 4 = 2 (two 16-byte stores per row)
    // For half  (2 bytes): VECTOR_SIZE = 8 -> VECTORS_PER_ROW = 8 / 8 = 1 (one 16-byte store per row)
    constexpr int VECTORS_PER_ROW = THREAD_TILE_N / VECTOR_SIZE; 

    // 6. Direct Vectorized Store Loop from Registers to Global Memory
    #pragma unroll
    for (int r = 0; r < THREAD_TILE_M; ++r) {
        int globalRow = rowStart + r;

        // Row-level boundary check
        if (globalRow < M) {

            #pragma unroll
            for (int v = 0; v < VECTORS_PER_ROW; ++v) {
                int localCol = v * VECTOR_SIZE;
                int globalCol = colStart + localCol;

                // Address pointer to store location in Matrix C
                const T* src_ptr = &accum[r][localCol];
                T* dst_ptr       = &C[globalRow * N + globalCol];

                // FAST PATH: Entire 16-byte vector fits in N AND memory address is 16-byte aligned
                if ((globalCol + VECTOR_SIZE <= N) && 
                    (reinterpret_cast<uintptr_t>(dst_ptr) % 16 == 0)) 
                {
                    *reinterpret_cast<uint4*>(dst_ptr) = 
                        *reinterpret_cast<const uint4*>(src_ptr);
                } 
                // SAFE PATH: Scalar fallback for boundary edges or unaligned leading dimensions
                else {
                    #pragma unroll
                    for (int j = 0; j < VECTOR_SIZE; ++j) {
                        if (globalCol + j < N) {
                            C[globalRow * N + globalCol + j] = accum[r][localCol + j];
                        }
                    }
                }
            }

        }
    }
}

template<typename T>
__global__ void matmul_kernel_3stage_cpasync(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int K, int N
)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK = 16;
    constexpr int STAGES = 3; // 3-Stage Multi-Buffering
    
    constexpr int WARP_ROWS = 4;
    constexpr int WARP_COLS = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS;  // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS;  // 64
    
    constexpr int GROUPS = 4;
    constexpr int LANES_PER_GROUP = 8;
    
    constexpr int THREAD_TILE_M = 8;
    constexpr int THREAD_TILE_N = 8;
    
    constexpr int UINT4_BYTES = 16;
    constexpr int VECTOR_SIZE = UINT4_BYTES / sizeof(T);
    constexpr int SMEM_PAD_A = VECTOR_SIZE; 

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    int warpRow = warpId / WARP_COLS;
    int warpCol = warpId % WARP_COLS;
    
    int groupId = laneId / LANES_PER_GROUP;
    int laneInGroup = laneId % LANES_PER_GROUP;
    
    int threadRow = warpRow * WARP_TILE_M + groupId * THREAD_TILE_M;
    int threadCol = warpCol * WARP_TILE_N + laneInGroup * THREAD_TILE_N;
    
    // ========== 3-STAGE SHARED MEMORY RING BUFFER ==========
    __shared__ T tileA[STAGES][BLOCK_M][BK];
    __shared__ T tileB[STAGES][BK][BLOCK_N];
    // =======================================================
    
    T accum[THREAD_TILE_M][THREAD_TILE_N] = {};
    T A_reg[THREAD_TILE_M];
    T B_reg[THREAD_TILE_N];
    
    int loadId = tid * VECTOR_SIZE;

    auto load_A = [&](int stage_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = blockIdx.y * BLOCK_M + row;
            int globalCol = current_k_tile + col;
            
            bool valid = (globalRow < M) && (globalCol < K);
            int remaining_bytes = (K - globalCol) * static_cast<int>(sizeof(T));
            int swizzled_col = swizzle_col_A<T>(row, col);
            
            cp_async_16(
                &tileA[stage_idx][row][swizzled_col],
                &A[globalRow * K + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    auto load_B = [&](int stage_idx, int current_k_tile) {
        #pragma unroll
        for(int i = loadId; i < BK * BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRow = current_k_tile + row;
            int globalCol = blockIdx.x * BLOCK_N + col;
            
            bool valid = (globalRow < K) && (globalCol < N);
            int remaining_bytes = (N - globalCol) * static_cast<int>(sizeof(T));
            
            cp_async_16(
                &tileB[stage_idx][row][col],
                &B[globalRow * N + globalCol],
                valid,
                remaining_bytes
            );
        }
    };

    int write_stage = 0;
    int read_stage  = 0;

    // ========== PROLOGUE: PREFETCH (STAGES - 1) TILES ==========
    #pragma unroll
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        int k_tile = stage * BK;
        if (k_tile < K) {
            load_A(write_stage, k_tile);
            load_B(write_stage, k_tile);
            cp_async_commit();
            write_stage = (write_stage + 1) % STAGES;
        }
    }

    // ========== MAIN LOOP: 3-STAGE PIPELINE ==========
    for (int tile = 0; tile < K; tile += BK)
    {
        // 1. Issue async load for upcoming tile (tile + 2 * BK)
        int fetch_tile = tile + (STAGES - 1) * BK;
        if (fetch_tile < K) {
            load_A(write_stage, fetch_tile);
            load_B(write_stage, fetch_tile);
            cp_async_commit();
            write_stage = (write_stage + 1) % STAGES;
        }

        // 2. Wait until current read_stage completes.
        // wait_group<STAGES - 2>() leaves (STAGES - 2) groups pending in flight,
        // allowing future loads to keep running while guaranteeing read_stage is ready.
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        // 3. Compute current tile from read_stage
        read_stage = tile % STAGES;
        #pragma unroll
        for (int k = 0; k < BK; k++)
        {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                int swizzled_k = swizzle_col_A<T>(threadRow + i, k);
                A_reg[i] = tileA[read_stage][threadRow + i][swizzled_k];
            }
            
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_N; i++) {
                B_reg[i] = tileB[read_stage][k][threadCol + i];
            }
            
            #pragma unroll
            for (int r = 0; r < THREAD_TILE_M; r++) {
                #pragma unroll
                for (int c = 0; c < THREAD_TILE_N; c++) {
                    accum[r][c] += A_reg[r] * B_reg[c];
                }
            }
        }

    }

    // Drain any remaining async copies
    cp_async_wait<0>();

    // ========== VECTORIZED EPILOGUE ==========
    constexpr int VECTORS_PER_ROW = THREAD_TILE_N / VECTOR_SIZE; 
    int rowStart = blockIdx.y * BLOCK_M + threadRow;
    int colStart = blockIdx.x * BLOCK_N + threadCol;

    #pragma unroll
    for (int r = 0; r < THREAD_TILE_M; ++r) {
        int globalRow = rowStart + r;
        if (globalRow < M) {
            #pragma unroll
            for (int v = 0; v < VECTORS_PER_ROW; ++v) {
                int localCol = v * VECTOR_SIZE;
                int globalCol = colStart + localCol;

                const T* src_ptr = &accum[r][localCol];
                T* dst_ptr       = &C[globalRow * N + globalCol];

                if ((globalCol + VECTOR_SIZE <= N) && 
                    (reinterpret_cast<uintptr_t>(dst_ptr) % 16 == 0)) 
                {
                    *reinterpret_cast<uint4*>(dst_ptr) = 
                        *reinterpret_cast<const uint4*>(src_ptr);
                } else {
                    #pragma unroll
                    for (int j = 0; j < VECTOR_SIZE; ++j) {
                        if (globalCol + j < N) {
                            C[globalRow * N + globalCol + j] = accum[r][localCol + j];
                        }
                    }
                }
            }
        }
    }
}


// ========== TENSOR CORE MATMUL: 3-STAGE CP.ASYNC + WMMA ==========
// Combines the 3-stage asynchronous pipeline from matmul_kernel_3stage_cpasync
// with WMMA tensor core instructions (nvcuda::wmma).
//
// Tiling layout (per block):
//   BLOCK_M x BLOCK_N = 128 x 128, BK = 16 (WMMA K-tile dimension)
//   8 warps arranged as WARP_ROWS(4) x WARP_COLS(2)
//   Each warp owns a WARP_TILE_M(32) x WARP_TILE_N(64) output subtile
//   decomposed into WARP_WMMA_M(2) x WARP_WMMA_N(4) = 8 WMMA fragments
//
// Pipeline:
//   Prologue  : prefetch (STAGES-1) tiles asynchronously into ring buffer
//   Main loop : issue next async load -> cp_async_wait<STAGES-2> ->
//               WMMA compute on current ring-buffer stage
//   Drain     : cp_async_wait<0> after loop exits
//
// Constraints:
//   Requires SM >= 7.0 (Volta) for WMMA, SM >= 8.0 for cp.async.
//   For best results pad M and N to multiples of 16 before calling.
template<typename InputType = half, typename AccType = float>
__global__ void matmul_kernel_tensor_core(
    const InputType* __restrict__ A,
    const InputType* __restrict__ B,
    AccType*         __restrict__ C,
    int M, int K, int N
)
{
    using namespace nvcuda;

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK      = 16;
    constexpr int STAGES  = 3;

    constexpr int WMMA_M  = 16;
    constexpr int WMMA_N  = 16;
    constexpr int WMMA_K  = 16;

    constexpr int WARP_ROWS   = 4;
    constexpr int WARP_COLS   = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS;       // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS;       // 64
    constexpr int WARP_WMMA_M = WARP_TILE_M / WMMA_M;      // 2
    constexpr int WARP_WMMA_N = WARP_TILE_N / WMMA_N;      // 4

    // 16-byte vector copies via cp.async.
    // For half (2 bytes): 8 elements per transfer.
    constexpr int VECTOR_BYTES = 16;
    constexpr int VECTOR_SIZE  = VECTOR_BYTES / static_cast<int>(sizeof(InputType));

    int tid    = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;   // unused directly; needed by WMMA implicitly

    int warpRow = warpId / WARP_COLS;   // 0..3
    int warpCol = warpId % WARP_COLS;   // 0..1

    // ========== 3-STAGE SHARED MEMORY RING BUFFER ==========
    // tileA: [STAGES][BLOCK_M rows][BK cols]  -- A sub-tile, K-major
    // tileB: [STAGES][BK rows][BLOCK_N cols]  -- B sub-tile, N-major
    // WMMA requires 16-byte aligned row starts.
    // half x BK=16 => 32 bytes/row (A), half x BLOCK_N=128 => 256 bytes/row (B) -- both ≥16B aligned.
    __shared__ __align__(16) InputType tileA[STAGES][BLOCK_M][BK];
    __shared__ __align__(16) InputType tileB[STAGES][BK][BLOCK_N];
    // =======================================================

    // Per-warp WMMA accumulator fragments: 2 row tiles x 4 col tiles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccType>
        acc[WARP_WMMA_M][WARP_WMMA_N];

#pragma unroll
    for (int i = 0; i < WARP_WMMA_M; i++)
#pragma unroll
        for (int j = 0; j < WARP_WMMA_N; j++)
            wmma::fill_fragment(acc[i][j], AccType(0));

    int loadId = tid * VECTOR_SIZE;

    // -------- async load helpers --------

    auto load_A = [&](int stage_idx, int k_tile) {
#pragma unroll
        for (int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
            int row       = i / BK;
            int col       = i % BK;
            int globalRow = blockIdx.y * BLOCK_M + row;
            int globalCol = k_tile + col;

            bool valid       = (globalRow < M) && (globalCol < K);
            int  remaining   = (K - globalCol) * static_cast<int>(sizeof(InputType));

            cp_async_16(
                &tileA[stage_idx][row][col],
                &A[globalRow * K + globalCol],
                valid,
                remaining
            );
        }
    };

    auto load_B = [&](int stage_idx, int k_tile) {
#pragma unroll
        for (int i = loadId; i < BK * BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
            int row       = i / BLOCK_N;
            int col       = i % BLOCK_N;
            int globalRow = k_tile + row;
            int globalCol = blockIdx.x * BLOCK_N + col;

            bool valid     = (globalRow < K) && (globalCol < N);
            int  remaining = (N - globalCol) * static_cast<int>(sizeof(InputType));

            cp_async_16(
                &tileB[stage_idx][row][col],
                &B[globalRow * N + globalCol],
                valid,
                remaining
            );
        }
    };

    int write_stage = 0;

    // ========== PROLOGUE: PREFETCH (STAGES - 1) TILES ==========
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        int k_tile = s * BK;
        if (k_tile < K) {
            load_A(write_stage, k_tile);
            load_B(write_stage, k_tile);
            cp_async_commit();
            write_stage = (write_stage + 1) % STAGES;
        }
    }

    // ========== MAIN LOOP: 3-STAGE PIPELINE + TENSOR CORE COMPUTE ==========
    for (int tile = 0; tile < K; tile += BK)
    {
        // 1. Issue async prefetch for the tile two stages ahead
        int fetch_tile = tile + (STAGES - 1) * BK;
        if (fetch_tile < K) {
            load_A(write_stage, fetch_tile);
            load_B(write_stage, fetch_tile);
            cp_async_commit();
            write_stage = (write_stage + 1) % STAGES;
        }

        // 2. Wait until the tile we are about to consume has landed in SMEM.
        //    Leaving (STAGES-2)=1 groups in-flight keeps the pipeline full.
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        // 3. WMMA compute: each warp processes its WARP_TILE_M x WARP_TILE_N subtile.
        int read_stage = (tile / BK) % STAGES;

        // Shared memory base pointers for this warp's view of A and B.
        // A is [BLOCK_M][BK] in shared; warp starts at row warpRow * WARP_TILE_M.
        const InputType* warpA =
            &tileA[read_stage][warpRow * WARP_TILE_M][0];

        // B is [BK][BLOCK_N] in shared; warp starts at col warpCol * WARP_TILE_N.
        const InputType* warpB =
            &tileB[read_stage][0][warpCol * WARP_TILE_N];

        // Inner WMMA: accumulate across the BK=16 depth dimension (one mma per (i,j) pair).
        wmma::fragment<
            wmma::matrix_b,
            WMMA_M, WMMA_N, WMMA_K,
            InputType,
            wmma::row_major
        > b_frag[WARP_WMMA_N];

        // Load all B fragments once per warp before reusing them for each A tile.
#pragma unroll
        for (int j = 0; j < WARP_WMMA_N; j++) {
            wmma::load_matrix_sync(b_frag[j], warpB + j * WMMA_N, BLOCK_N);
        }

#pragma unroll
        for (int i = 0; i < WARP_WMMA_M; i++) {
            wmma::fragment<
                wmma::matrix_a,
                WMMA_M, WMMA_N, WMMA_K,
                InputType,
                wmma::row_major
            > a_frag;

            // Row offset: i * WMMA_M * BK elements in the flat [BLOCK_M][BK] array
            wmma::load_matrix_sync(a_frag, warpA + i * WMMA_M * BK, BK);

#pragma unroll
            for (int j = 0; j < WARP_WMMA_N; j++) {
                wmma::mma_sync(acc[i][j], a_frag, b_frag[j], acc[i][j]);
            }
        }
    }

    // Drain any remaining in-flight async copies before exiting
    cp_async_wait<0>();
    __syncthreads();

    // ========== STORE ACCUMULATORS TO GLOBAL MEMORY ==========
    // Each warp writes its 2x4 grid of 16x16 output tiles.
    int cRow = blockIdx.y * BLOCK_M + warpRow * WARP_TILE_M;
    int cCol = blockIdx.x * BLOCK_N + warpCol * WARP_TILE_N;

#pragma unroll
    for (int i = 0; i < WARP_WMMA_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_WMMA_N; j++) {
            int outRow = cRow + i * WMMA_M;
            int outCol = cCol + j * WMMA_N;

            // Guard: only store tiles that lie fully within the output matrix.
            // Callers should pad M and N to multiples of WMMA_M/WMMA_N when
            // boundary-exact results are needed.
            if (outRow + WMMA_M <= M && outCol + WMMA_N <= N) {
                wmma::store_matrix_sync(
                    C + outRow * N + outCol,
                    acc[i][j],
                    N,
                    wmma::mem_row_major
                );
            }
        }
    }
}
// ============================================================


// ========== AMPERE TENSOR CORE MATMUL: 3-STAGE CP.ASYNC + ldmatrix + mma.sync ==========
// Replaces the WMMA abstraction in matmul_kernel_tensor_core with direct Ampere SM80 PTX.
// Same threadblock tiling, warp mapping, shared memory layout, and cp.async pipeline.
//
// Tiling layout (per block):
//   BLOCK_M x BLOCK_N = 128 x 128, BK = 16 (= MMA_K, no inner K-loop needed)
//   8 warps arranged as WARP_ROWS(4) x WARP_COLS(2)
//   Each warp owns a WARP_TILE_M(32) x WARP_TILE_N(64) output subtile
//   decomposed into WARP_MMA_M(2) x WARP_MMA_N(8) mma.sync.m16n8k16 operations
//
// PTX instructions used:
//   ldmatrix.sync.aligned.m8n8.x4.shared.b16       -- load 16x16 A tile into 4 regs
//   ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 -- load 16x8 B tile transposed (col-major)
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
// Requires SM >= 8.0 (Ampere). Do NOT use on Turing or older.
__global__ void matmul_kernel_ampere(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N
)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BK      = 16;
    constexpr int STAGES  = 3;

    // mma.sync.aligned.m16n8k16: MxNxK = 16x8x16
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;

    constexpr int WARP_ROWS   = 4;
    constexpr int WARP_COLS   = 2;
    constexpr int WARP_TILE_M = BLOCK_M / WARP_ROWS;    // 32
    constexpr int WARP_TILE_N = BLOCK_N / WARP_COLS;    // 64
    constexpr int WARP_MMA_M  = WARP_TILE_M / MMA_M;    // 2
    constexpr int WARP_MMA_N  = WARP_TILE_N / MMA_N;    // 8

    constexpr int VECTOR_BYTES = 16;
    constexpr int VECTOR_SIZE  = VECTOR_BYTES / static_cast<int>(sizeof(half));  // 8

    int tid    = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int warpRow = warpId / WARP_COLS;   // 0..3
    int warpCol = warpId % WARP_COLS;   // 0..1

    // ========== 3-STAGE SHARED MEMORY RING BUFFER ==========
    __shared__ __align__(16) half tileA[STAGES][BLOCK_M][BK];
    __shared__ __align__(16) half tileB[STAGES][BK][BLOCK_N];

    // ========== ACCUMULATORS ==========
    // mma.sync.m16n8k16 D layout: thread laneId holds 4 fp32 values mapping to
    //   d[0] = C[laneId/4 + 0][2*(laneId%4) + 0]
    //   d[1] = C[laneId/4 + 0][2*(laneId%4) + 1]
    //   d[2] = C[laneId/4 + 8][2*(laneId%4) + 0]
    //   d[3] = C[laneId/4 + 8][2*(laneId%4) + 1]
    float acc[WARP_MMA_M][WARP_MMA_N][4];
#pragma unroll
    for (int i = 0; i < WARP_MMA_M; i++)
#pragma unroll
        for (int j = 0; j < WARP_MMA_N; j++) {
            acc[i][j][0] = 0.f; acc[i][j][1] = 0.f;
            acc[i][j][2] = 0.f; acc[i][j][3] = 0.f;
        }

    int loadId = tid * VECTOR_SIZE;

    // -------- async load helpers (identical to matmul_kernel_tensor_core) --------
    auto load_A = [&](int stage_idx, int k_tile) {
#pragma unroll
        for (int i = loadId; i < BLOCK_M * BK; i += blockDim.x * VECTOR_SIZE) {
            int row       = i / BK;
            int col       = i % BK;
            int globalRow = blockIdx.y * BLOCK_M + row;
            int globalCol = k_tile + col;
            bool valid    = (globalRow < M) && (globalCol < K);
            int  remaining = (K - globalCol) * static_cast<int>(sizeof(half));
            cp_async_16(&tileA[stage_idx][row][col], &A[globalRow * K + globalCol], valid, remaining);
        }
    };

    auto load_B = [&](int stage_idx, int k_tile) {
#pragma unroll
        for (int i = loadId; i < BK * BLOCK_N; i += blockDim.x * VECTOR_SIZE) {
            int row        = i / BLOCK_N;
            int col        = i % BLOCK_N;
            int globalRow  = k_tile + row;
            int globalCol  = blockIdx.x * BLOCK_N + col;
            bool valid     = (globalRow < K) && (globalCol < N);
            int  remaining = (N - globalCol) * static_cast<int>(sizeof(half));
            cp_async_16(&tileB[stage_idx][row][col], &B[globalRow * N + globalCol], valid, remaining);
        }
    };

    int write_stage = 0;

    // ========== PROLOGUE: PREFETCH (STAGES - 1) TILES ==========
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        int k_tile = s * BK;
        if (k_tile < K) {
            load_A(write_stage, k_tile);
            load_B(write_stage, k_tile);
            cp_async_commit();
            write_stage = (write_stage + 1) % STAGES;
        }
    }

    // ========== MAIN LOOP: 3-STAGE PIPELINE + AMPERE TENSOR CORE COMPUTE ==========
    for (int tile = 0; tile < K; tile += BK)
    {
        // 1. Issue async prefetch for the tile two stages ahead.
        int fetch_tile = tile + (STAGES - 1) * BK;
        if (fetch_tile < K) {
            load_A(write_stage, fetch_tile);
            load_B(write_stage, fetch_tile);
            cp_async_commit();
            write_stage = (write_stage + 1) % STAGES;
        }

        // 2. Wait for the current tile's data to arrive in SMEM.
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        int read_stage = (tile / BK) % STAGES;

        // Warp's shared memory base pointers.
        const half* warpA = &tileA[read_stage][warpRow * WARP_TILE_M][0];
        const half* warpB = &tileB[read_stage][0][warpCol * WARP_TILE_N];

        // ---- Load A fragments via ldmatrix.x4 ----
        // For each i-th 16x16 sub-tile of A:
        //   Thread laneId provides row ptr at (laneId%8 + (laneId>>4)*8) within the tile,
        //   at column ((laneId>>3)&1)*8.  Four sub-tiles are loaded atomically into 4 regs.
        uint32_t a_frag[WARP_MMA_M][4];
#pragma unroll
        for (int i = 0; i < WARP_MMA_M; i++) {
            int a_row = (laneId % 16) + i * MMA_M;
            int a_col = (laneId / 16) * 8;
            uint32_t smem_ptr = __cvta_generic_to_shared(&warpA[a_row * BK + a_col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(a_frag[i][0]), "=r"(a_frag[i][1]),
                  "=r"(a_frag[i][2]), "=r"(a_frag[i][3])
                : "r"(smem_ptr)
            );
        }

        // ---- Load B fragments via ldmatrix.x2.trans ----
        // For each j-th 16x8 sub-tile of B (row-major in SMEM):
        //   Thread laneId provides a row pointer at row (laneId%16), col j*MMA_N.
        //   The .trans qualifier delivers data in col-major register layout as required
        //   by mma.sync.m16n8k16.row.col's B operand (2 .b32 regs = 4 fp16 per thread).
        uint32_t b_frag[WARP_MMA_N][2];
#pragma unroll
        for (int j = 0; j < WARP_MMA_N; j++) {
            int b_row = laneId % 16;
            int b_col = j * MMA_N;
            uint32_t smem_ptr = __cvta_generic_to_shared(&warpB[b_row * BLOCK_N + b_col]);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b_frag[j][0]), "=r"(b_frag[j][1])
                : "r"(smem_ptr)
            );
        }

        // ---- mma.sync.aligned.m16n8k16.row.col compute ----
#pragma unroll
        for (int i = 0; i < WARP_MMA_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_MMA_N; j++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, "
                    "{%4,%5,%6,%7}, "
                    "{%8,%9}, "
                    "{%10,%11,%12,%13};\n"
                    : "=f"(acc[i][j][0]), "=f"(acc[i][j][1]),
                      "=f"(acc[i][j][2]), "=f"(acc[i][j][3])
                    : "r"(a_frag[i][0]), "r"(a_frag[i][1]),
                      "r"(a_frag[i][2]), "r"(a_frag[i][3]),
                      "r"(b_frag[j][0]), "r"(b_frag[j][1]),
                      "f"(acc[i][j][0]), "f"(acc[i][j][1]),
                      "f"(acc[i][j][2]), "f"(acc[i][j][3])
                );
            }
        }
    }

    cp_async_wait<0>();
    __syncthreads();

    // ========== STORE ACCUMULATORS TO GLOBAL MEMORY ==========
    // mma.sync.m16n8k16 D-register layout per mma tile (16 rows x 8 cols):
    //   thread laneId owns: row0=laneId/4, row1=laneId/4+8, col0=2*(laneId%4), col1=col0+1
    int cRow = blockIdx.y * BLOCK_M + warpRow * WARP_TILE_M;
    int cCol = blockIdx.x * BLOCK_N + warpCol * WARP_TILE_N;

    int frag_r0 = laneId / 4;
    int frag_r1 = laneId / 4 + 8;
    int frag_c0 = (laneId % 4) * 2;
    int frag_c1 = frag_c0 + 1;

#pragma unroll
    for (int i = 0; i < WARP_MMA_M; i++) {
        int rowBase = cRow + i * MMA_M;
#pragma unroll
        for (int j = 0; j < WARP_MMA_N; j++) {
            int colBase = cCol + j * MMA_N;

            int r0 = rowBase + frag_r0;
            int r1 = rowBase + frag_r1;
            int c0 = colBase + frag_c0;
            int c1 = colBase + frag_c1;

            if (r0 < M && c0 < N) C[r0 * N + c0] = acc[i][j][0];
            if (r0 < M && c1 < N) C[r0 * N + c1] = acc[i][j][1];
            if (r1 < M && c0 < N) C[r1 * N + c0] = acc[i][j][2];
            if (r1 < M && c1 < N) C[r1 * N + c1] = acc[i][j][3];
        }
    }
}
// ============================================================


namespace {
    int grid_size(int64_t size) {
        int threads = 256;
        return static_cast<int>((size + threads - 1) / threads);
    }
}

// C++ overloads for add
void cuda_add(const float* a, const float* b, float* out, int64_t size) {
    add_kernel<float><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_add(const int32_t* a, const int32_t* b, int32_t* out, int64_t size) {
    add_kernel<int32_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_add(const int64_t* a, const int64_t* b, int64_t* out, int64_t size) {
    add_kernel<int64_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_add(const float16_t* a, const float16_t* b, float16_t* out, int64_t size) {
    add_kernel<float16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_add(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size) {
    add_kernel<bfloat16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

// C++ overloads for sub
void cuda_sub(const float* a, const float* b, float* out, int64_t size) {
    sub_kernel<float><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_sub(const int32_t* a, const int32_t* b, int32_t* out, int64_t size) {
    sub_kernel<int32_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_sub(const int64_t* a, const int64_t* b, int64_t* out, int64_t size) {
    sub_kernel<int64_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_sub(const float16_t* a, const float16_t* b, float16_t* out, int64_t size) {
    sub_kernel<float16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_sub(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size) {
    sub_kernel<bfloat16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

// C++ overloads for mul
void cuda_mul(const float* a, const float* b, float* out, int64_t size) {
    mul_kernel<float><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_mul(const int32_t* a, const int32_t* b, int32_t* out, int64_t size) {
    mul_kernel<int32_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_mul(const int64_t* a, const int64_t* b, int64_t* out, int64_t size) {
    mul_kernel<int64_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_mul(const float16_t* a, const float16_t* b, float16_t* out, int64_t size) {
    mul_kernel<float16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_mul(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size) {
    mul_kernel<bfloat16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

// C++ overloads for div
void cuda_div(const float* a, const float* b, float* out, int64_t size) {
    div_kernel<float><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_div(const int32_t* a, const int32_t* b, int32_t* out, int64_t size) {
    div_kernel<int32_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_div(const int64_t* a, const int64_t* b, int64_t* out, int64_t size) {
    div_kernel<int64_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_div(const float16_t* a, const float16_t* b, float16_t* out, int64_t size) {
    div_kernel<float16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

void cuda_div(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size) {
    div_kernel<bfloat16_t><<<grid_size(size), 256>>>(a, b, out, size);
}

// C++ overloads for relu
void cuda_relu(const float* input, float* out, int64_t size) {
    relu_kernel<float><<<grid_size(size), 256>>>(input, out, size);
}

void cuda_relu(const float16_t* input, float16_t* out, int64_t size) {
    relu_kernel<float16_t><<<grid_size(size), 256>>>(input, out, size);
}

void cuda_relu(const bfloat16_t* input, bfloat16_t* out, int64_t size) {
    relu_kernel<bfloat16_t><<<grid_size(size), 256>>>(input, out, size);
}

// C++ overloads for matmul (float only for now)
void cuda_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_kernel<float><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_shared_memory(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel_shared_memory<float><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_register_blocking(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matmul_kernel_register_blcoking<float><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_vectorized_input(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matmul_kernel_vectorized_input<float><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_warp_tiling(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_warp_tiling<float><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_double_buffered(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_double_buffered<float><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_double_buffered_cpasync(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_double_buffered_cpasync<float><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_double_buffered_swizzled(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_double_buffered_swizzled<float><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_vector_storage(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_vector_storage<float><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_3stage_cpasync(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_3stage_cpasync<float><<<grid, block>>>(A, B, C, M, K, N);
}

// FP16 non-tensor-core matmul launchers
void cuda_matmul(const half* A, const half* B, half* C, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_kernel<half><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_shared_memory(const half* A, const half* B, half* C, int M, int K, int N) {
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel_shared_memory<half><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_register_blocking(const half* A, const half* B, half* C, int M, int K, int N) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matmul_kernel_register_blcoking<half><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_vectorized_input(const half* A, const half* B, half* C, int M, int K, int N) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matmul_kernel_vectorized_input<half><<<blocks, threads>>>(A, B, C, M, K, N);
}

void cuda_matmul_warp_tiling(const half* A, const half* B, half* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_warp_tiling<half><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_double_buffered(const half* A, const half* B, half* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_double_buffered<half><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_double_buffered_cpasync(const half* A, const half* B, half* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_double_buffered_cpasync<half><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_double_buffered_swizzled(const half* A, const half* B, half* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_double_buffered_swizzled<half><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_vector_storage(const half* A, const half* B, half* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_vector_storage<half><<<grid, block>>>(A, B, C, M, K, N);
}

void cuda_matmul_3stage_cpasync(const half* A, const half* B, half* C, int M, int K, int N) {
    dim3 block(256, 1, 1);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matmul_kernel_3stage_cpasync<half><<<grid, block>>>(A, B, C, M, K, N);
}

void launch_matmul_tensor_core(
    const half* A, const half* B, float* C,
    int M, int K, int N
)
{
    int M_pad = ((M + 15) / 16) * 16;
    int K_pad = ((K + 15) / 16) * 16;
    int N_pad = ((N + 15) / 16) * 16;

    // Fast path: all dimensions are already WMMA-aligned.
    if (M_pad == M && K_pad == K && N_pad == N) {
        dim3 block(256, 1, 1);
        dim3 grid(
            (N + 128 - 1) / 128,
            (M + 128 - 1) / 128
        );

        matmul_kernel_tensor_core<half, float>
            <<<grid, block>>>(A, B, C, M, K, N);

        cudaDeviceSynchronize();
        return;
    }

    // Padded path: zero-pad A, B, and C so the kernel can operate on full tiles.
    half*  A_pad = nullptr;
    half*  B_pad = nullptr;
    float* C_pad = nullptr;

    cudaMalloc(&A_pad, static_cast<size_t>(M_pad) * K_pad * sizeof(half));
    cudaMalloc(&B_pad, static_cast<size_t>(K_pad) * N_pad * sizeof(half));
    cudaMalloc(&C_pad, static_cast<size_t>(M_pad) * N_pad * sizeof(float));

    cudaMemsetAsync(A_pad, 0, static_cast<size_t>(M_pad) * K_pad * sizeof(half), 0);
    cudaMemsetAsync(B_pad, 0, static_cast<size_t>(K_pad) * N_pad * sizeof(half), 0);
    cudaMemsetAsync(C_pad, 0, static_cast<size_t>(M_pad) * N_pad * sizeof(float), 0);

    // Place the original matrices in the top-left of the padded buffers.
    cudaMemcpy2DAsync(
        A_pad, K_pad * sizeof(half),
        A,     K * sizeof(half),
        K * sizeof(half), M,
        cudaMemcpyDeviceToDevice, 0
    );

    cudaMemcpy2DAsync(
        B_pad, N_pad * sizeof(half),
        B,     N * sizeof(half),
        N * sizeof(half), K,
        cudaMemcpyDeviceToDevice, 0
    );

    dim3 block(256, 1, 1);
    dim3 grid(
        (N_pad + 128 - 1) / 128,
        (M_pad + 128 - 1) / 128
    );

    matmul_kernel_tensor_core<half, float>
        <<<grid, block>>>(A_pad, B_pad, C_pad, M_pad, K_pad, N_pad);

    // Copy only the valid M x N region from the padded output back to C.
    cudaMemcpy2DAsync(
        C,     N * sizeof(float),
        C_pad, N_pad * sizeof(float),
        N * sizeof(float), M,
        cudaMemcpyDeviceToDevice, 0
    );

    cudaDeviceSynchronize();

    cudaFree(A_pad);
    cudaFree(B_pad);
    cudaFree(C_pad);
}

void launch_matmul_ampere(
    const half* A, const half* B, float* C,
    int M, int K, int N
)
{
    int M_pad = ((M + 15) / 16) * 16;
    int K_pad = ((K + 15) / 16) * 16;
    int N_pad = ((N + 15) / 16) * 16;

    dim3 block(256, 1, 1);

    // Fast path: all dimensions are already aligned.
    if (M_pad == M && K_pad == K && N_pad == N) {
        dim3 grid(
            (N + 128 - 1) / 128,
            (M + 128 - 1) / 128
        );
        matmul_kernel_ampere<<<grid, block>>>(A, B, C, M, K, N);
        cudaDeviceSynchronize();
        return;
    }

    // Padded path: zero-pad A, B, and C so the kernel always operates on full tiles.
    half*  A_pad = nullptr;
    half*  B_pad = nullptr;
    float* C_pad = nullptr;

    cudaMalloc(&A_pad, static_cast<size_t>(M_pad) * K_pad * sizeof(half));
    cudaMalloc(&B_pad, static_cast<size_t>(K_pad) * N_pad * sizeof(half));
    cudaMalloc(&C_pad, static_cast<size_t>(M_pad) * N_pad * sizeof(float));

    cudaMemsetAsync(A_pad, 0, static_cast<size_t>(M_pad) * K_pad * sizeof(half), 0);
    cudaMemsetAsync(B_pad, 0, static_cast<size_t>(K_pad) * N_pad * sizeof(half), 0);
    cudaMemsetAsync(C_pad, 0, static_cast<size_t>(M_pad) * N_pad * sizeof(float), 0);

    cudaMemcpy2DAsync(
        A_pad, K_pad * sizeof(half),
        A,     K * sizeof(half),
        K * sizeof(half), M,
        cudaMemcpyDeviceToDevice, 0
    );

    cudaMemcpy2DAsync(
        B_pad, N_pad * sizeof(half),
        B,     N * sizeof(half),
        N * sizeof(half), K,
        cudaMemcpyDeviceToDevice, 0
    );

    dim3 grid(
        (N_pad + 128 - 1) / 128,
        (M_pad + 128 - 1) / 128
    );

    matmul_kernel_ampere<<<grid, block>>>(A_pad, B_pad, C_pad, M_pad, K_pad, N_pad);

    cudaMemcpy2DAsync(
        C,     N * sizeof(float),
        C_pad, N_pad * sizeof(float),
        N * sizeof(float), M,
        cudaMemcpyDeviceToDevice, 0
    );

    cudaDeviceSynchronize();

    cudaFree(A_pad);
    cudaFree(B_pad);
    cudaFree(C_pad);
}

// ---------------------------------------------------------------------------
// Softmax on the last dimension.
// Treats `x` as contiguous [outer_size, dim_size] row-major and computes
// softmax for each row in parallel.  Input and output are device pointers.
// ---------------------------------------------------------------------------
__global__ void softmax_forward_kernel(const float* x, float* out,
                                       int64_t outer_size, int64_t dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size) return;

    const float* x_row = x + idx * dim_size;
    float* out_row = out + idx * dim_size;

    float max_val = -__int_as_float(0x7f800000);
    for (int64_t i = 0; i < dim_size; ++i) {
        max_val = fmaxf(max_val, x_row[i]);
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < dim_size; ++i) {
        float v = expf(x_row[i] - max_val);
        out_row[i] = v;
        sum += v;
    }

    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < dim_size; ++i) {
        out_row[i] *= inv_sum;
    }
}

__global__ void softmax_backward_kernel(const float* s, const float* dy, float* dx,
                                        int64_t outer_size, int64_t dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size) return;

    const float* s_row  = s  + idx * dim_size;
    const float* dy_row = dy + idx * dim_size;
    float* dx_row = dx + idx * dim_size;

    float dot = 0.0f;
    for (int64_t i = 0; i < dim_size; ++i) {
        dot += dy_row[i] * s_row[i];
    }

    for (int64_t i = 0; i < dim_size; ++i) {
        dx_row[i] = s_row[i] * (dy_row[i] - dot);
    }
}

void cuda_softmax_forward(const float* x, float* out,
                          int64_t outer_size, int64_t dim_size) {
    int threads = 256;
    int blocks = static_cast<int>((outer_size + threads - 1) / threads);
    softmax_forward_kernel<<<blocks, threads>>>(x, out, outer_size, dim_size);
}

void cuda_softmax_backward(const float* s, const float* dy, float* out,
                           int64_t outer_size, int64_t dim_size) {
    int threads = 256;
    int blocks = static_cast<int>((outer_size + threads - 1) / threads);
    softmax_backward_kernel<<<blocks, threads>>>(s, dy, out, outer_size, dim_size);
}

__global__ void fill_value_kernel(float* data, float value, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

void cuda_fill(float* data, float value, int64_t size) {
    int threads = 256;
    int blocks = static_cast<int>((size + threads - 1) / threads);
    fill_value_kernel<<<blocks, threads>>>(data, value, size);
}

// ---------------------------------------------------------------------------
// LayerNorm on the last dimension.
// ---------------------------------------------------------------------------
__global__ void layernorm_forward_kernel(const float* x, const float* gamma, const float* beta,
                                         float* out, int batch, int D, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    const float* xrow = x + idx * D;
    float* outrow = out + idx * D;

    float mean = 0.0f;
    for (int j = 0; j < D; ++j) mean += xrow[j];
    mean /= D;

    float var = 0.0f;
    for (int j = 0; j < D; ++j) {
        float d = xrow[j] - mean;
        var += d * d;
    }
    var /= D;
    float inv_std = rsqrtf(var + eps);

    for (int j = 0; j < D; ++j) {
        float normalized = (xrow[j] - mean) * inv_std;
        outrow[j] = normalized * gamma[j] + beta[j];
    }
}

__global__ void layernorm_backward_kernel(const float* x, const float* gamma, const float* dy,
                                          float* dx, float* dg, float* db,
                                          int batch, int D, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    const float* xrow  = x  + idx * D;
    const float* dyrow = dy + idx * D;
    float* dxrow = dx + idx * D;

    float mean = 0.0f;
    for (int j = 0; j < D; ++j) mean += xrow[j];
    mean /= D;

    float var = 0.0f;
    for (int j = 0; j < D; ++j) {
        float d = xrow[j] - mean;
        var += d * d;
    }
    var /= D;
    float inv_std = rsqrtf(var + eps);

    // Accumulate d_gamma and d_beta across the batch using global atomics.
    for (int j = 0; j < D; ++j) {
        float x_hat = (xrow[j] - mean) * inv_std;
        atomicAdd(&dg[j], dyrow[j] * x_hat);
        atomicAdd(&db[j], dyrow[j]);
    }

    // Per-row means for the dx formula.
    float mean_g_dy = 0.0f;
    float mean_g_dy_xhat = 0.0f;
    for (int j = 0; j < D; ++j) {
        float x_hat = (xrow[j] - mean) * inv_std;
        float g_dy = gamma[j] * dyrow[j];
        mean_g_dy += g_dy;
        mean_g_dy_xhat += g_dy * x_hat;
    }
    mean_g_dy /= D;
    mean_g_dy_xhat /= D;

    for (int j = 0; j < D; ++j) {
        float x_hat = (xrow[j] - mean) * inv_std;
        dxrow[j] = (gamma[j] * dyrow[j] - mean_g_dy - x_hat * mean_g_dy_xhat) * inv_std;
    }
}

void cuda_layernorm_forward(const float* x, const float* gamma, const float* beta,
                            float* out, int batch, int D, float eps) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    layernorm_forward_kernel<<<blocks, threads>>>(x, gamma, beta, out, batch, D, eps);
}

void cuda_layernorm_backward(const float* x, const float* gamma, const float* dy,
                             float* dx, float* dg, float* db,
                             int batch, int D, float eps) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    layernorm_backward_kernel<<<blocks, threads>>>(x, gamma, dy, dx, dg, db, batch, D, eps);
}

// ---------------------------------------------------------------------------
// Embedding lookup and embedding_grad.
// ---------------------------------------------------------------------------
__global__ void embedding_forward_kernel(const int64_t* indices, const float* weight,
                                         float* out, int total_indices, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_indices * D) return;
    int i = idx / D;
    int j = idx % D;
    int64_t emb_idx = indices[i];
    out[idx] = weight[emb_idx * D + j];
}

__global__ void embedding_backward_kernel(const int64_t* indices, const float* grad_out,
                                          float* grad_weight, int total_indices, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_indices * D) return;
    int i = idx / D;
    int j = idx % D;
    int64_t emb_idx = indices[i];
    atomicAdd(&grad_weight[emb_idx * D + j], grad_out[idx]);
}

void cuda_embedding_forward(const int64_t* indices, const float* weight,
                            float* out, int total_indices, int D) {
    int threads = 256;
    int blocks = (total_indices * D + threads - 1) / threads;
    embedding_forward_kernel<<<blocks, threads>>>(indices, weight, out, total_indices, D);
}

void cuda_embedding_backward(const int64_t* indices, const float* grad_out,
                             float* grad_weight, int total_indices, int D) {
    int threads = 256;
    int blocks = (total_indices * D + threads - 1) / threads;
    embedding_backward_kernel<<<blocks, threads>>>(indices, grad_out, grad_weight, total_indices, D);
}

// ---------------------------------------------------------------------------
// Cross-entropy loss over 3D logits [B, T, V] and 2D int64 targets [B, T].
// ---------------------------------------------------------------------------
__global__ void crossentropy_batched_forward_kernel(const float* logits,
                                                    const int64_t* targets,
                                                    float* loss,
                                                    int B, int T, int V) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    if (n >= N) return;

    const float* row = logits + n * V;
    int target_class = static_cast<int>(targets[n]);

    float max_logit = row[0];
    for (int v = 1; v < V; ++v) {
        max_logit = fmaxf(max_logit, row[v]);
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < V; ++v) {
        sum_exp += expf(row[v] - max_logit);
    }

    float logit_target = row[target_class];
    float row_loss = -logit_target + max_logit + logf(sum_exp);
    atomicAdd(loss, row_loss);
}

__global__ void crossentropy_batched_backward_kernel(const float* logits,
                                                     const int64_t* targets,
                                                     float* grad,
                                                     float scale,
                                                     int B, int T, int V) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    if (n >= N) return;

    const float* row = logits + n * V;
    float* grad_row = grad + n * V;
    int target_class = static_cast<int>(targets[n]);

    float max_logit = row[0];
    for (int v = 1; v < V; ++v) {
        max_logit = fmaxf(max_logit, row[v]);
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < V; ++v) {
        sum_exp += expf(row[v] - max_logit);
    }

    for (int v = 0; v < V; ++v) {
        float softmax_val = expf(row[v] - max_logit) / sum_exp;
        float one_hot = (v == target_class) ? 1.0f : 0.0f;
        grad_row[v] = (softmax_val - one_hot) * scale;
    }
}

void cuda_crossentropy_batched_forward(const float* logits, const int64_t* targets,
                                       float* loss, int B, int T, int V) {
    int N = B * T;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    crossentropy_batched_forward_kernel<<<blocks, threads>>>(logits, targets, loss, B, T, V);
}

void cuda_crossentropy_batched_backward(const float* logits, const int64_t* targets,
                                        float* grad, float scale,
                                        int B, int T, int V) {
    int N = B * T;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    crossentropy_batched_backward_kernel<<<blocks, threads>>>(logits, targets, grad, scale, B, T, V);
}

// ---------------------------------------------------------------------------
// GELU forward and backward (tanh approximation).
// ---------------------------------------------------------------------------
__global__ void gelu_forward_kernel(const float* input, float* output, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float x3 = x * x * x;
    output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
}

__global__ void gelu_backward_kernel(const float* input, const float* grad_output,
                                     float* grad_input, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = input[idx];
    float dy = grad_output[idx];
    float c = 0.7978845608f;
    float k = 0.044715f;
    float x_cubed = x * x * x;
    float z = c * (x + k * x_cubed);
    float t = tanhf(z);
    float sech_sq = 1.0f - t * t;
    float inner = c * (1.0f + 3.0f * k * x * x);
    float gelu_grad = 0.5f * (1.0f + t) * (1.0f + x * sech_sq * inner);
    grad_input[idx] = dy * gelu_grad;
}

void cuda_gelu(const float* input, float* output, int64_t size) {
    int threads = 256;
    int blocks = static_cast<int>((size + threads - 1) / threads);
    gelu_forward_kernel<<<blocks, threads>>>(input, output, size);
}

void cuda_gelu_backward(const float* input, const float* grad_output,
                        float* grad_input, int64_t size) {
    int threads = 256;
    int blocks = static_cast<int>((size + threads - 1) / threads);
    gelu_backward_kernel<<<blocks, threads>>>(input, grad_output, grad_input, size);
}

// ---------------------------------------------------------------------------
// Reductions: full sum and sum over a single dimension (up to 8 dimensions).
// ---------------------------------------------------------------------------
__global__ void reduce_sum_kernel(const float* input, float* output, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    atomicAdd(output, input[idx]);
}

__global__ void sum_dim_kernel(const float* input, float* output,
                               const int64_t* in_shape,
                               const int64_t* in_strides,
                               const int64_t* out_shape,
                               const int64_t* out_strides,
                               int64_t dim, int64_t ndim,
                               int64_t out_numel, int64_t dim_size) {
    int64_t out_flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_flat >= out_numel) return;

    int64_t out_indices[8];
    int64_t temp = out_flat;
    int64_t ndim_out = ndim - 1;
    for (int64_t i = ndim_out - 1; i >= 0; --i) {
        out_indices[i] = temp % out_shape[i];
        temp /= out_shape[i];
    }

    int64_t base = 0;
    for (int64_t d = 0; d < dim; ++d) {
        base += out_indices[d] * in_strides[d];
    }
    for (int64_t d = dim; d < ndim_out; ++d) {
        base += out_indices[d] * in_strides[d + 1];
    }

    float sum = 0.0f;
    int64_t stride = in_strides[dim];
    for (int64_t i = 0; i < dim_size; ++i) {
        sum += input[base + i * stride];
    }
    output[out_flat] = sum;
}

void cuda_reduce_sum(const float* input, float* output, int64_t size) {
    int threads = 256;
    int blocks = static_cast<int>((size + threads - 1) / threads);
    reduce_sum_kernel<<<blocks, threads>>>(input, output, size);
}

void cuda_sum_dim(const float* input, float* output,
                  const int64_t* in_shape, const int64_t* in_strides,
                  const int64_t* out_shape, const int64_t* out_strides,
                  int64_t dim, int64_t ndim, int64_t out_numel, int64_t dim_size) {
    int threads = 256;
    int blocks = static_cast<int>((out_numel + threads - 1) / threads);
    sum_dim_kernel<<<blocks, threads>>>(input, output,
                                        in_shape, in_strides,
                                        out_shape, out_strides,
                                        dim, ndim, out_numel, dim_size);
}