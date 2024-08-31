#define BLOCK_SIZE 4        //根据GPU调整
#define MAX_CHANNELS 32

__kernel void Conv2d_prelu(
        __global float* data,
        __global const float* weight,
        __global float* out,
        unsigned int out_size,
        unsigned char ks,
        unsigned char stride,
        unsigned char with_bias,
        short in_h,
        short in_w,
        short in_c,
        short out_h,
        short out_w,
        short out_c
        )
{
    // 这里定义了固定大小的局部内存
    __local float local_data[BLOCK_SIZE * BLOCK_SIZE * MAX_CHANNELS];

    // 其他代码部分保持不变
    unsigned int idx = get_global_id(0);
    size_t global_items_sum = get_global_size(0);
    short pad_value = 0;
    char half_ks = (ks -1)/2;
    char mid_kernel_idx = half_ks * ks + half_ks;

    unsigned int correspond_x, correspond_y, correspond_idx;
    int img_size = in_w * in_h;

    int kernel_pos;
    float *weight_out;
    float weight_value;
    short c, y, x;
    float result;

    for (unsigned int i = idx; i < out_size; i += global_items_sum)
    {
        result = 0;

        c = i % out_c;
        x = (i % (out_c * out_w)) / out_c;
        y = i / (out_c * out_w);

        // 将需要的输入数据加载到局部内存中
        for (short in_channel = 0; in_channel < in_c; in_channel++) {
            correspond_x = stride * x - half_ks;
            correspond_y = stride * y - half_ks;
            correspond_idx = in_channel + correspond_y * in_w * in_c + correspond_x * in_c;

            if(correspond_y >= 0 && correspond_y < in_h && correspond_x >= 0 && correspond_x < in_w) {
                local_data[get_local_id(0) * in_c + in_channel] = data[correspond_idx];
            } else {
                local_data[get_local_id(0) * in_c + in_channel] = pad_value;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // 同步以确保所有线程加载数据完成

        for (char offset_y = -half_ks; offset_y <= half_ks; offset_y++) {
            for (char offset_x = -half_ks; offset_x <= half_ks; offset_x++) {
                for (short in_channel = 0; in_channel < in_c; in_channel++) {
                    kernel_pos = mid_kernel_idx + offset_y * ks + offset_x;
                    weight_out = weight + c * in_c * ks * ks;
                    weight_value = weight_out[in_channel * ks * ks + kernel_pos];

                    result += weight_value * local_data[(y + offset_y) * in_c * BLOCK_SIZE + (x + offset_x) * in_c + in_channel];
                }
            }
        }

        out[i] = result;
        if(with_bias){
            out[i] += weight[(out_c) * in_c * ks * ks + c];
        }

        if (out[i] < 0) {
            out[i] = weight[(out_c) * in_c * ks * ks + with_bias * out_c+c] * out[i];
        }
    }
}
