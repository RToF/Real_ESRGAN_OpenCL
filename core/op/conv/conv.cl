//单batch，卷积长宽相等，pad默认规范
__kernel void Conv2d(
        __global float* data,
        __global const float* weight,
        __global float* out,
                unsigned int out_size,
                unsigned char ks,                        // kernel_size
                unsigned char stride,
                unsigned char with_bias,                 // 不让用bool
                short in_h,
                short in_w,
                short in_c,
                short out_h,
                short out_w,
                short out_c )
{
    unsigned int idx = get_global_id(0);
    if (idx >= out_size) return;                    // 超出的直接返回
    size_t global_items_sum = get_global_size(0);   // 总共的items数量
    //printf("global_items_sum:%d\n",global_items_sum);

    //pad实际上没有用
    short pad_value = 0;

    char half_ks = (ks -1)/2;                       // 必须是整数
    char mid_kernel_idx = half_ks * ks + half_ks;   // 卷积核在整个卷积中的索引位置(单通道)

    // 存放对应在输入图像上的索引位置
    int correspond_x, correspond_y, correspond_idx;
    int img_size = in_w * in_h;

    // 卷积核位置和值的计算
    int kernel_pos;
    float *weight_out;
    float weight_value;

    for (unsigned int i = idx; i < out_size; i += global_items_sum) // 若是总的item数量少于像素数，就要负责多个
    {
        // 所对应在输出图上负责的像素位置（每个item负责一个输出图上的像素）
        short c, y, x;
        x = i % out_w;
        y = i % (out_w * out_h) / out_w;
        c = i / (out_w * out_h);


        // 存放单个item负责区域的最终卷积结果
        float result=0;

        // 取到每个值周围 (ks-1)/2的数字组成一排，超出的部分就是pad
        // 从列到行，这里假设卷积核的长宽相等，且为奇数
        for (short in_channel = 0; in_channel < in_c; in_channel++)
        {
            for (char offset_y = -half_ks; offset_y <= half_ks; offset_y++)
            {
                for (char offset_x = -half_ks; offset_x <= half_ks; offset_x++)
                {
                    // 卷积核的哪一个位置
                    kernel_pos = mid_kernel_idx + offset_y * ks + offset_x;
                    // 对应输出通道的起始位置
                    weight_out = weight + c * in_c * ks * ks;
                    // 对应哪一个输入通道的卷积核
                    weight_value = weight_out[in_channel * ks * ks + kernel_pos];

                    // 所对应在输入图像的索引(卷积核中心)
                    correspond_x = stride * x;
                    correspond_y = stride * y;
                    correspond_idx = in_channel * img_size + correspond_y * in_w + correspond_x;

                    if( correspond_y + offset_y < 0 || correspond_y + offset_y >= in_h) { result += pad_value * weight_value; continue;}  // 判断是否是pad部分
                    if( correspond_x + offset_x < 0 || correspond_x + offset_x >= in_w) { result += pad_value * weight_value; continue;}  // 判断是否是pad部分

                    result += weight_value * data[correspond_idx + in_w * offset_y + offset_x];
                }
            }
        }
        out[i] = result; // 总共的weight数量+第几个输出通道的bias索引
        if(with_bias){
            //printf("%f\n",weight[(out_c) * in_c * ks * ks + c]);
            out[i] += weight[(out_c) * in_c * ks * ks + c];
        }
    }
}
