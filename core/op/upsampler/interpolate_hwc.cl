

__kernel void Interpolate(  __global float* input,
                       __global float* output,
                       unsigned int out_size,
                       short in_h,
                       short in_w,
                       short in_c,
                       short out_h,
                       short out_w,
                       short out_c,
                       unsigned char scale
){
    unsigned int idx = get_global_id(0);
    size_t global_items_sum = get_global_size(0);

    // 对应输出的哪一个位置
    short c, y, x;

    // 对应输入的哪一个位置
    short cor_c, cor_y, cor_x;
    unsigned int cor_idx; //防止溢出！！！！

    for (unsigned int i = idx; i < out_size; i += global_items_sum){ // 若是总的item数量少于像素数，就要负责多个

        c = idx % out_c;
        y = idx / (out_c * out_w);
        x = (idx % (out_c * out_w)) / out_c;

        cor_x = x / scale;
        cor_y = y / scale;
        cor_c = c;  // 只有这里和PixelShuffle不一样
        cor_idx = cor_c + cor_x * in_c + cor_y * in_c * in_w;

        output[i] = input[cor_idx];
    }


}