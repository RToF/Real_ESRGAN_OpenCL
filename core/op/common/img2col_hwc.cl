// 合并读 + 合并写
// 输出还是HWC格式 （行主序）
__kernel void img2col(  __global float* input,
                        __global float* output,
                        short in_h,
                        short in_w,
                        short in_c,
                        unsigned char ks,                        // kernel_size
                        unsigned char stride,
                        unsigned char with_bias
){
    unsigned int idx = get_global_id(0);
    size_t global_items_sum = get_global_size(0);

    char padding = (ks - 1) / 2;
    float pad_value = 0;
    unsigned int out_size = in_h * in_w * in_c;

    // 计算img2col每一个特征图变成矩阵以后的大小
    short out_h = (in_h - ks + 2 * padding) / stride + 1;
    short out_w = (in_w - ks + 2 * padding) / stride + 1;

    char half_ks = (ks -1)/2;
    if(!with_bias){
        // 每一个线程都搬1个，一个位置一个位置搬，因为这样子是先搬同一个通道的，内存连续
        for (unsigned int i = idx; i < out_h * out_w * ks * ks * in_c; i += global_items_sum){

            // -------------- 计算在img2col上负责的位置 --------------- //
            unsigned int x, y, c;   // 写short容易越界
            c = i % in_c;
            y = i / (in_c * ks * ks);                //第几行
            x = (i % (in_c * ks * ks)) / in_c;       //第几列

            // ------------------ 对应在输入上的位置 ------------------ //
            short correspond_x, correspond_y, correspond_c;
            char offset_x, offset_y;

            // 1.根据在第几列得出是卷积核的相对偏移
            offset_x = x % ks - half_ks;
            offset_y = x / ks - half_ks;

            // 2.根据在第几行得出卷积核中心位置
            correspond_x = y % out_w;
            correspond_y = y / out_w;

            // 3.得到最终对应的位置
            correspond_x = correspond_x * stride + offset_x;
            correspond_y = correspond_y * stride + offset_y;
            correspond_c = c;

            // ---------------------- 搬运 ------------------------ //
            // 防止越界
            if( correspond_y < 0 || correspond_y >= in_h) { output[i] = pad_value; continue;}  // 判断是否是pad部分
            if( correspond_x < 0 || correspond_x >= in_w) { output[i] = pad_value; continue;}  // 判断是否是pad部分

            output[i] = input[correspond_c + correspond_x * in_c + correspond_y * in_w * in_c];
            //printf("[%d,%d] -> [%d,%d]", x,y,correspond_x, correspond_y);
        }
    }
    else{
        // 每一个线程都搬1个，一个位置一个位置搬，因为这样子是先搬同一个通道的，内存连续

        for (unsigned int i = idx; i < out_h * out_w * (in_c * ks * ks+1); i += global_items_sum){
            if( ((i+1) % (in_c * ks * ks + 1 )) == 0 ){
                output[i] = 1;
            }
            else{
                // -------------- 计算在img2col上负责的位置 --------------- //
                unsigned int x, y, c;                   // 写short容易越界
                c = i % in_c;
                y = i / (in_c * ks * ks + 1);                //第几行
                x = (i % (in_c * ks * ks + 1)) / in_c;       //第几列

                // ------------------ 对应在输入上的位置 ------------------ //
                short correspond_x, correspond_y, correspond_c;
                char offset_x, offset_y;

                // 1.根据在第几列得出是卷积核的相对偏移
                offset_x = x % ks - half_ks;
                offset_y = x / ks - half_ks;

                // 2.根据在第几行得出卷积核中心位置
                correspond_x = y % out_w;
                correspond_y = y / out_w;

                // 3.得到最终对应的位置
                correspond_x = correspond_x * stride + offset_x;
                correspond_y = correspond_y * stride + offset_y;
                correspond_c = c;

                // ---------------------- 搬运 ------------------------ //
                // 防止越界
                if( correspond_y < 0 || correspond_y >= in_h) { output[i] = pad_value; continue;}  // 判断是否是pad部分
                if( correspond_x < 0 || correspond_x >= in_w) { output[i] = pad_value; continue;}  // 判断是否是pad部分

                output[i] = input[correspond_c + correspond_x * in_c + correspond_y * in_w * in_c];
                //printf("[%d,%d] -> [%d,%d]", x,y,correspond_x, correspond_y);
            }

        }
    }

}