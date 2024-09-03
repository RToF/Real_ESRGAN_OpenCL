// 合并读 + 合并写
// 输出还是HWC格式 (列主序)
// TODO:初始化的时候计算一次，放入__constant
__kernel void weight2col(  __global float* input,
                           __global float* output,
                           short in_c,
                           short out_c,
                           unsigned char ks,
                           unsigned char with_bias
){
    unsigned int idx = get_global_id(0);
    size_t global_items_sum = get_global_size(0);
    unsigned int x, y;
    unsigned int in_idx;
    short cor_ker_x, cor_ker_y, cor_in_c, cor_out_c;
    if (!with_bias){
        unsigned int out_size = ks * in_c * ks * out_c;             // 每列 (ks * in_c) * ks, 每行 out_c
        // 每一个线程都搬1个，一个位置一个位置搬，因为这样子是先搬同一个通道的，内存连续
        for (unsigned int i = idx; i < out_size; i += global_items_sum){

            // ------------- 计算在weight2col上负责的位置 ------------- //


            y = i % (in_c * ks * ks);       //第几行
            x = i / (in_c * ks * ks);       //第几列

            // ------------------ 对应在输入上的位置 ------------------ //



            // 1.根据在第几列得到是在哪一个输出通道
            cor_out_c = x;

            // 2.根据在第几行得出具体哪一个输入通道  (同一个通道挨着放)
            cor_in_c = y % in_c;

            // 3.根据在第几行得出在卷积核上的坐标
            cor_ker_x = (y % (in_c * ks)) / in_c;
            cor_ker_y = y / (in_c * ks);

            // 4.得到最终对应的位置
            in_idx = cor_out_c * in_c * ks * ks + cor_in_c * ks * ks + cor_ker_y * ks  + cor_ker_x;  // weight是直接从torch权重读取的，所以是NCHW的!!!

            // ----------------------- 搬运 ------------------------- //

            output[i] = input[in_idx];

            //printf(" [%d,%d,%d]: %f   ", cor_in_c, cor_ker_y, cor_ker_x, input[in_idx]);
        }
    }
    else{
        unsigned int out_size = ks * in_c * ks * out_c + out_c;             // 每列 (ks * in_c) * ks+1, 每行 out_c
        for (unsigned int i = idx; i < out_size; i += global_items_sum){

            if( ((i+1) % (ks * in_c * ks+1)) == 0 ){
               x = i / (ks * in_c * ks + 1);            // 看是第几个输出通道的
               output[i] = input[out_c * ks * ks * in_c + x];
            }
            else{
                // ------------- 计算在weight2col上负责的位置 ------------- //

                y = i % (in_c * ks * ks + 1);       //第几行
                x = i / (in_c * ks * ks + 1);       //第几列

                // ------------------ 对应在输入上的位置 ------------------ //


                // 1.根据在第几列得到是在哪一个输出通道
                cor_out_c = x;

                // 2.根据在第几行得出具体哪一个输入通道  (同一个通道挨着放)
                cor_in_c = y % in_c;

                // 3.根据在第几行得出在卷积核上的坐标
                cor_ker_x = (y % (in_c * ks)) / in_c;
                cor_ker_y = y / (in_c * ks);

                // 4.得到最终对应的位置

                in_idx = cor_out_c * in_c * ks * ks + cor_in_c * ks * ks + cor_ker_y * ks  + cor_ker_x;  // weight是直接从torch权重读取的，所以是NCHW的!!!

                // ----------------------- 搬运 ------------------------- //

                output[i] = input[in_idx];

                //printf(" [%d,%d,%d]: %f   ", cor_in_c, cor_ker_y, cor_ker_x, input[in_idx]);
            }


        }
    }
}