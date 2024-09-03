__kernel void gemm( __global float* A,        // data
                    __global float* B,        // weight
                    __global float* output,   // M*N
                    unsigned int M,
                    unsigned int N,
                    unsigned int K
){
    // TODO:在编译的时候传入
    const char tile_size = 16;  //要与外部的groups_max_item协调

    size_t global_sum_col = get_global_size(0);
    size_t global_sum_row = get_global_size(1);

    __local float A_tile[tile_size * tile_size];
    __local float B_tile[tile_size * tile_size];

    unsigned int local_col = get_local_id(0);
    unsigned int local_row = get_local_id(1);

    // 计算填充后的尺寸, 不然local里面有些部分没有item搬运
    unsigned int pad_N = (N/tile_size + 1) * tile_size;
    unsigned int pad_M = (M/tile_size + 1) * tile_size;

    float result = 0.0f;
    unsigned int tile_num = K / tile_size;

    for (unsigned int global_row = get_global_id(1); global_row < pad_M; global_row += global_sum_row){
        for (unsigned int global_col = get_global_id(0); global_col < pad_N; global_col += global_sum_col){

            result = 0.0f;

            for (unsigned int k = 0; k < tile_num; k++){
                // 边界检查，防止越界访问
                if (global_row < M && k * tile_size + local_col < K) {
                    A_tile[local_row * tile_size + local_col] = A[global_row * K + k * tile_size + local_col];
                }

                if (global_col < N && k * tile_size + local_row < K) {
                    B_tile[local_row * tile_size + local_col] = B[global_col * K + k * tile_size + local_row];
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                for(unsigned int m = 0; m < tile_size; m++){
                    result += A_tile[local_row * tile_size + m] * B_tile[m * tile_size + local_col];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // 处理最后剩下的部分
            if(tile_num * tile_size + local_col < K) {
                if (global_row < M) {
                    A_tile[local_row * tile_size + local_col] = A[global_row * K + tile_num * tile_size + local_col];
                }
            }

            if(tile_num * tile_size + local_row < K) {
                if (global_col < N) {
                    B_tile[local_row * tile_size + local_col] = B[global_col * K + tile_num * tile_size + local_row];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            for(unsigned int k = 0; k < K - tile_num * tile_size; k++){
                result += A_tile[local_row * tile_size + k] * B_tile[k * tile_size + local_col];
            }

            // 边界检查，防止写入超出范围
            if (global_row < M && global_col < N){
                output[global_row * N + global_col] = result;
            }

        }
    }
}