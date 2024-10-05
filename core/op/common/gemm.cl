
// TODO:还没加上prelu
__kernel void gemm( __global float* A,        // data
                    __global float* B,        // weight
                    __global float* output,   // M*N
                    unsigned int M,
                    unsigned int N,
                    unsigned int K
){
    // TODO:在编译的时候传入
    const char tile_size = 32;  //要与外部的groups_max_item协调

    size_t global_sum_col = get_global_size(0);
    size_t global_sum_row = get_global_size(1);

    __local float A_tile[tile_size][tile_size];
    __local float B_tile[tile_size][tile_size];

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
                if (global_row < M ) {
                    A_tile[local_row][local_col] = A[global_row * K + k * tile_size + local_col];
                }

                if (global_col < N ) {
                    B_tile[local_row][local_col]  = B[global_col * K + k * tile_size + local_row];
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                for(unsigned int m = 0; m < tile_size; m++){
                    result += A_tile[local_row][m]  * B_tile[m][local_col] ;
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // 处理边界部分
            if(tile_num * tile_size + local_col < K) {
                if (global_row < M) {
                    A_tile[local_row][local_col] = A[global_row * K + tile_num * tile_size + local_col];
                }
            }

            if(tile_num * tile_size + local_row < K) {
                if (global_col < N) {
                    B_tile[local_row][local_col] = B[global_col * K + tile_num * tile_size + local_row];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            for(unsigned int k = 0; k < K - tile_num * tile_size; k++){
                result += A_tile[local_row][k] * B_tile[k][local_col];
            }

            // 边界检查，防止写入超出范围
            if (global_row < M && global_col < N){
                output[global_row * N + global_col] = result;
            }

        }
    }
}


//// 确保block中每个线程可以读取一行，数量刚好和块的行/列相等
//#define group_size_row 64          // 要与外部的groups_max_item协调
//#define group_size_col 64          // 要与外部的groups_max_item协调
//#define tile_size_k 4          // 要与外部的groups_max_item协调
//#define item_ts_col 8      // item_tile_size
//#define item_ts_row 8
//
//// 每个group内的每个item负责一个小区域，并存储到寄存器中
//__kernel void gemm( __global float* A,        // data
//                    __global float* B,        // weight
//                    __global float* output,   // M*N
//                    unsigned int M,
//                    unsigned int N,
//                    unsigned int K
//){
//    unsigned char tile_col_block = group_size_col / item_ts_col;    // block的一行多少个tile
//    unsigned char tile_row_block = group_size_row / item_ts_row;
//
//    // TODO: 合并访问 + bank_conflict
//    __local float A_tile[group_size_row][tile_size_k];
//    __local float B_tile[tile_size_k][group_size_col];
//
//    unsigned int group_id_col = get_group_id(0);
//    unsigned int group_id_row = get_group_id(1);
//    unsigned int local_col = get_local_id(0);
//    unsigned int local_row = get_local_id(1);
//    unsigned int group_num_col = get_num_groups(0);
//    unsigned int group_num_row = get_num_groups(1);
//
//    unsigned int loop_num = K / tile_size_k;
//
//    // item在group的索引,每个item读一行
//    short tile_per_row_group = group_size_col / item_ts_col;
//    short iid = local_row * tile_per_row_group + local_col;
//
//    for(unsigned roll_row = group_id_row; roll_row < ((M + group_size_row - 1) / group_size_row); roll_row+= group_num_row)
//        for(unsigned roll_col = group_id_col; roll_col < ((N + group_size_col - 1) / group_size_col); roll_col+= group_num_col){
//
//            float a_reg[item_ts_row];
//            float b_reg[item_ts_col];
//            float c_reg[item_ts_row][item_ts_col] = {0.0f};
//
//            // 得到每一个item搬运数据到__local时 要负责的行/列
//            unsigned int res_row_a = roll_row * group_size_row + iid;
//            unsigned int res_col_b = roll_col * group_size_col + iid;
//
//            // item在groups内要负责的区域索引
//            short row_start = local_row * item_ts_row; //得到tile在第几行/起始行索引
//            short col_start = local_col * item_ts_col;
//
//            // 得到每一个item搬运数据到寄存器时 要负责的行/列的起始索引
//            unsigned int global_row = roll_row * group_size_row + row_start;
//            unsigned int global_col = roll_col * group_size_col + col_start;
//            // 表示距离边界M/N的距离
//            int row_bound = (global_row + item_ts_row<M) ? 0:(global_row + item_ts_row - M); // 可能为负数,不需要再额外+1变成标量，此时已经是标量不是索引
//            int col_bound = (global_col + item_ts_col<N) ? 0:(global_col + item_ts_col - N);
//            // -------------------------- 对规整区域处理 ---------------------------//
//            for (unsigned int k = 0; k < loop_num; k++) {
//                // 读取A
//                if (res_row_a < M) {
//                    for(unsigned char i = 0; i < tile_size_k; i++)
//                        A_tile[iid][i] = A[res_row_a * K + k * tile_size_k + i];
//                }
//                // 读取B
//                if (res_col_b < N) {
//                    for(unsigned char i = 0; i < tile_size_k; i++)
//                        B_tile[i][iid] = B[res_col_b * K + k * tile_size_k + i];
//                }
//
//                barrier(CLK_LOCAL_MEM_FENCE);
//                // 放入到寄存器中
//                for(unsigned char t = 0; t < tile_size_k; t++){
//                    // 存入a_reg
//                    for(int ta = 0; ta < item_ts_row ; ta++){
//                        a_reg[ta] = A_tile[row_start + ta][t];
//                    }
//                    // 存入b_reg
//                    for(int tb = 0; tb < item_ts_col ; tb++){
//                        b_reg[tb] = B_tile[t][col_start + tb];
//                    }
//                    // TODO:非常耗时 -> 这里把 -row_bound去掉，-col_bound也去掉加速明显???是编译器自动向量化了吗
//                    for(int ta = 0; ta < item_ts_row ; ta++)
//                        for(int tb = 0; tb < item_ts_col ; tb++){
//                            c_reg[ta][tb] += a_reg[ta] * b_reg[tb];
//                    }
//                }
//                barrier(CLK_LOCAL_MEM_FENCE);
//            }
//            // -------------------------- 对边界区域处理 ---------------------------//
//            // 计算距离边界的距离, 相当于现在的tile_size_k为l
//            char l = K - loop_num * tile_size_k;
//
//            // 读取A
//            if (res_row_a < M) {
//                for(int i = 0; i < l; i++)
//                    A_tile[iid][i] = A[res_row_a * K + loop_num * tile_size_k + i];
//            }
//            // 读取B
//            if (res_col_b < N) {
//                for(int i = 0; i < l; i++)
//                    B_tile[i][iid] = B[res_col_b * K + loop_num * tile_size_k + i];
//            }
//            barrier(CLK_LOCAL_MEM_FENCE);
//            // 放入到寄存器中
//            for(int t = 0; t < l; t++){
//                // 存入a_reg
//                for(int ta = 0; ta < item_ts_row ; ta++){
//                    a_reg[ta] = A_tile[row_start + ta][t];
//                }
//                // 存入b_reg
//                for(int tb = 0; tb < item_ts_col ; tb++){
//                    b_reg[tb] = B_tile[t][col_start + tb];
//                }
//                // 计算
//                for(int ta = 0; ta < item_ts_row ; ta++)
//                    for(int tb = 0; tb < item_ts_col ; tb++){
//                        c_reg[ta][tb] += a_reg[ta] * b_reg[tb];
//                }
//            }
//            barrier(CLK_LOCAL_MEM_FENCE);
//            // -------------------------- 存入到output ---------------------------//
//            for(int ta = 0; ta < item_ts_row - row_bound; ta++){
//                int cur_row = global_row + ta;
//                for(int tb = 0; tb < item_ts_col - col_bound; tb++){
//                    int cur_col = global_col + tb;
//                    output[cur_row * N + cur_col] = c_reg[ta][tb];
//                }
//            }
//
//
//    }
//}
