//////////////////////////////////////////////////////////////////////////////////////
//////////////////////// 系数来源是 wincnn ，与原论文和NCNN不一致 /////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

__constant float AT[6][8] = {
        {1, 1,   1,     1,     1,     1,          1,          0},
        {0, 1,  -1,    2,    -2,     1.0/2,     -1.0/2,     0},
        {0, 1,   1,     4,     4,     1.0/4,      1.0/4,      0},
        {0, 1,  -1,    8,    -8,     1.0/8,     -1.0/8,     0},
        {0, 1,   1,    16,    16,     1.0/16,     1.0/16,    0},
        {0, 1,  -1,   32,  -32,     1.0/32,     -1.0/32,   1}
};

__constant float G[8][3] = {
        {1.0f,             0.0f,              0.0f       },
        {-2.0f/9,        -2.0f/9,        -2.0f/9   },
        {-2.0f/9,        2.0f/9,         -2.0f/9   },
        {1.0f/90,        1.0f/45,         2.0f/45  },
        {1.0f/90,        -1.0f/45,        2.0f/45  },
        {32.0f/45,       16.0f/45,        8.0f/45  },
        {32.0f/45,       -16.0f/45,       8.0f/45  },
        {0.0f,             0.0f,              1.0f       }
};
//__constant float BT[8][8] = {
//        {  1,           0,        -21.0/4,       0,       21.0/4,       0,        -1,  0 },
//        {  0,           1,           1,       -17.0/4,    -17.0/4,       1,       1,   0 },
//        {  0,          -1,           1,       17.0/4,     -17.0/4,      -1,       1,   0 },
//        {  0,        1.0/2,         1.0/4,    -5.0/2,     -5.0/4,       2,        1,   0 },
//        {  0,       -1.0/2,         1.0/4,    5.0/2,      -5.0/4,      -2,        1,   0 },
//        {  0,           2,           4,       -5.0/2,       -5,       1.0/2,      1,   0 },
//        {  0,          -2,           4,       5.0/2,        -5,       -1.0/2,     1,   0 },
//        {  0,          -1,           0,       21.0/4,        0,       -21.0/4,    0,   1 }
//};


#define ks 3
#define pad 1
/**
 *  输入输出都是CHW格式
 *  如果不够除以8的，则往右下角继续延伸，所以最后的尺寸就是下/右两条边进行约束
 **/

__kernel void GgGT( __global float* g,
                    __global float* output,
//                    unsigned int outsize,
                    int inc,
                    int outc
){
    const unsigned int gdx = get_group_id(0);
    const unsigned int total_groups = get_num_groups(0);

    const float t0 = 2.0f / 9;
    const float t1 = 1.0f / 45;
    const float t2 = 2.0f / 45;
    const float t3 = 1.0f / 90;
    const float t4 = 32.0f / 45;
    const float t5 = 16.0f / 45;
    const float t6 = 8.0f / 45;

    // 对每一个卷积核进行
    for(unsigned int i = gdx; i < inc * outc; i+= total_groups) {

        float gGT[ks][8] = {0.0f};
        float g0,g1,g2,g3,g4,g5,g6,g7,g8;
        float GgGT[8][8] = {0.0f};

        // 得到对应的起始位置
        unsigned int data = ks * ks * i;
        // 3x3载入寄存器中
        g0 = g[data + 0];
        g1 = g[data + 1];
        g2 = g[data + 2];
        g3 = g[data + 3];
        g4 = g[data + 4];
        g5 = g[data + 5];
        g6 = g[data + 6];
        g7 = g[data + 7];
        g8 = g[data + 8];
        // gGT
        gGT[0][0] = g0;
        gGT[0][1] = -t0*(g0 + g1 + g2);
        gGT[0][2] = -t0*(g0 - g1 + g2);
        gGT[0][3] = t3*g0 + t1*g1 + t2*g2;
        gGT[0][4] = t3*g0 - t1*g1 + t2*g2;
        gGT[0][5] = t4*g0 + t5*g1 + t6*g2;
        gGT[0][6] = t4*g0 - t5*g1 + t6*g2;
        gGT[0][7] = g2;

        gGT[1][0] = g3;
        gGT[1][1] = -t0*(g3 + g4 + g5);
        gGT[1][2] = -t0*(g3 - g4 + g5);
        gGT[1][3] = t3*g3 + t1*g4 + t2*g5;
        gGT[1][4] = t3*g3 - t1*g4 + t2*g5;
        gGT[1][5] = t4*g3 + t5*g4 + t6*g5;
        gGT[1][6] = t4*g3 - t5*g4 + t6*g5;
        gGT[1][7] = g5;

        gGT[2][0] = g6;
        gGT[2][1] = -t0*(g6 + g7 + g8);
        gGT[2][2] = -t0*(g6 - g7 + g8);
        gGT[2][3] = t3*g6 + t1*g7 + t2*g8;
        gGT[2][4] = t3*g6 - t1*g7 + t2*g8;
        gGT[2][5] = t4*g6 + t5*g7 + t6*g8;
        gGT[2][6] = t4*g6 - t5*g7 + t6*g8;
        gGT[2][7] = g8;

        // G(gGT)
        for (int r = 0; r < 8; r++){
            for (int c = 0; c < 8; c++){
                for (int t = 0; t < 3; t++)
                    GgGT[r][c] += G[r][t] * gGT[t][c];
            }
        }

        // 找到对应的输出位置
        unsigned int out = 8 * 8 * i;
        // 输出
        for(int r = 0; r < 8; r++){
            for(int c = 0; c < 8; c++) {
                output[out + 8*r + c] = GgGT[r][c];
            }
        }

    }
}


// 变成CHW格式，同一层的所有tiles挨着，也就是CTHW,输入还是HWC的
__kernel void img2tiles(__global float* d,        // data
                       __global float* output,
                       int in_h,
                       int in_w,
                       int in_c,
                       int tiles_w,
                       int tiles_h
){
    unsigned int idx = get_global_id(0);
    unsigned int total_items = get_global_size(0);
    unsigned int tiles_size = tiles_h * tiles_w;
    for(int i = idx; i < (in_c * tiles_size) << 6; i+=total_items){

        // 对应第几个tile
        unsigned int tdx = i >> 6;
        // 对应第几个通道
        unsigned int c = tdx / tiles_size;
        // 对应的位置
        unsigned int out_row = (i % 64) >> 3;
        unsigned int out_col = (i % 64) % 8;

        // 根据输出的位置确定输入
        unsigned int tile_w = tdx % tiles_size % tiles_w;
        unsigned int tile_h = tdx % tiles_size / tiles_w;

        unsigned int in_col_begin = -pad + tile_w * 6;  // 从-pad的索引开始计算，由于tile之间重叠2个，所以按6来倍增
        unsigned int in_row_begin = -pad + tile_h * 6;
        // 对应图片上的位置
        unsigned int cor_col = in_col_begin + out_col;
        unsigned int cor_row = in_row_begin + out_row;

        //越界处理
        if (cor_col < 0 || cor_col >= in_w) {output[i] = 0; continue;}
        if (cor_row < 0 || cor_row >= in_h) {output[i] = 0; continue;}

        output[i] = d[cor_row * in_w * in_c + cor_col * in_c + c];

    }

}

__kernel void BTdB( __global float *d,
                    __global float *output,
                    int in_c,
                    int tiles_num
){
    const unsigned int gid = get_group_id(0);
    const unsigned int total_groups = get_num_groups(0);
    const unsigned int lid = get_local_id(0);

    float t0,t1,t2,t3,t4,t5,t6,t7;
    float result[8] = {0.0f};

    const float c1 = 21.0f/4;
    const float c2 = 17.0f/4;
    const float c3 = 5.0f/2;
    const float c4 = 5.0f/4;
    const float c5 = 1.0f/2;
    const float c6 = 1.0f/4;
    const float c7 = 2.0f;
    const float c8 = 4.0f;
    const float c9 = 5.0f;

    float _temp1;
    float _temp2;

    __local float dB[8][8];
    for(int i = gid; i < tiles_num; i += total_groups){
        // ---------------- dB ---------------- //
        // 得到要负责的行 的起始地址
        unsigned int data_idx = (i << 6) + (lid << 3);
        // 每个item负责一行
        t0 = d[data_idx + 0];
        t1 = d[data_idx + 1];
        t2 = d[data_idx + 2];
        t3 = d[data_idx + 3];
        t4 = d[data_idx + 4];
        t5 = d[data_idx + 5];
        t6 = d[data_idx + 6];
        t7 = d[data_idx + 7];

        result[0] = t0 + c1*(t4-t2) - t6;
        result[7] = -t1 + c1*(t3-t5) + t7;

        _temp1 = t2 - c2*t4 + t6;
        _temp2 = t1 - c2*t3 + t5;
        result[1] = _temp1 + _temp2;
        result[2] = _temp1 - _temp2;

        _temp1 = c6*t2 - c4*t4 + t6;
        _temp2 = c5*t1 - c3*t3 + c7*t5;
        result[3] = _temp1 + _temp2;
        result[4] = _temp1 - _temp2;

        _temp1 = c8*t2 - c9*t4 + t6;
        _temp2 = c7*t1 - c3*t3 + c5*t5;
        result[5] = _temp1 + _temp2;
        result[6] = _temp1 - _temp2;

        // 得到dB,存入共享内存中
        for (int c = 0; c < 8; c++){
            dB[lid][c] = result[c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // ---------------- BTdB ---------------- //
        data_idx = (i << 6) + lid;
        // 此时每一个item负责一列
        t0 = dB[0][lid];
        t1 = dB[1][lid];
        t2 = dB[2][lid];
        t3 = dB[3][lid];
        t4 = dB[4][lid];
        t5 = dB[5][lid];
        t6 = dB[6][lid];
        t7 = dB[7][lid];
//        barrier(CLK_LOCAL_MEM_FENCE);

        output[data_idx + 0] = t0 + c1*(t4-t2) - t6;
        output[data_idx + 56] = -t1 + c1*(t3-t5) + t7;

        _temp1 = t2 - c2*t4 + t6;
        _temp2 = t1 - c2*t3 + t5;
        output[data_idx + 8] = _temp1 + _temp2;
        output[data_idx + 16] = _temp1 - _temp2;

        _temp1 = c6*t2 - c4*t4 + t6;
        _temp2 = c5*t1 - c3*t3 + c7*t5;
        output[data_idx + 24] = _temp1 + _temp2;
        output[data_idx + 32] = _temp1 - _temp2;

        _temp1 = c8*t2 - c9*t4 + t6;
        _temp2 = c7*t1 - c3*t3 + c5*t5;
        output[data_idx + 40] = _temp1 + _temp2;
        output[data_idx + 48] = _temp1 - _temp2;

    }

}


__kernel void UV( __global float *U,        // [n,c,8,8]
                  __global float *V,        // [c,t,8,8]
                  __global float *output,   // [n,c,t,8,8]
                  int inc,
                  int outc,
                  int tiles_per_map
){
    unsigned int gid = get_group_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int total_groups = get_num_groups(0);

    // 每一个groups负责 最终输出[n,c,t,8,8]中的一个tile
    for (int i = gid; i < tiles_per_map * inc * outc; i += total_groups){
        // 得到对应的输入通道和输出通道
        unsigned int t = i % tiles_per_map;
        unsigned int c = (i / tiles_per_map) % inc;
        unsigned int n = i / (tiles_per_map * inc);             // gid / (tiles_per_map * inc)
        // 得到对应的卷积核索引位置
        unsigned int k_idx = ((n * inc) << 6) + (c << 6);

        // 得到数值
        output[(i<<6)+lid] = U[k_idx+lid] * V[((c*tiles_per_map) << 6) + (t << 6) + lid];
    }


}

__kernel void ATMA( __global float *M,        // [n,c,t,8,8]
                    __global float *output,   // [n,c,t,6,6]
                    int inc,
                    int outc,
                    int tiles_per_map
){
    const unsigned int gid = get_group_id(0);
    const unsigned int total_groups = get_num_groups(0);
    const unsigned int lid = get_local_id(0);

    float t0,t1,t2,t3,t4,t5,t6,t7;
    float result[6];

    float c1 = 1.0f/2;
    float c2 = 2.0f;
    float c3 = 1.0f/4;
    float c4 = 4.0f;
    float c5 = 1.0f/8;
    float c6 = 8.0f;
    float c7 = 1.0f/16;
    float c8 = 16.0f;
    float c9 = 1.0f/32;
    float c10 = 32.0f;

    __local float temp_row[8];
    __local float MA[8][6];

    for (int i = gid; i < tiles_per_map * inc * outc; i += total_groups){
        // MA
        // 每个item负责一行
        unsigned int data_idx = (i << 6) + (lid << 3);
        t0 = M[data_idx + 0];
        t1 = M[data_idx + 1];
        t2 = M[data_idx + 2];
        t3 = M[data_idx + 3];
        t4 = M[data_idx + 4];
        t5 = M[data_idx + 5];
        t6 = M[data_idx + 6];
        t7 = M[data_idx + 7];

        result[0] = t0+t1+t2+t3+t4+t5+t6;
        result[1] = t1-t2+c2*(t3-t4)+c1*(t5-t6);
        result[2] = t1+t2+c4*(t3+t4)+c3*(t5+t6);
        result[3] = t1-t2+c6*(t3-t4)+c5*(t5-t6);
        result[4] = t1+t2+c8*(t3+t4)+c7*(t5+t6);
        result[5] = t1-t2+c10*(t3-t4)+c9*(t5-t6)+t7;

        for (int c = 0; c < 6; c++){
            MA[lid][c] = result[c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);   //TODO:每个groups不到32个item，所以只有一个warp，必定同步
        // AT(MA),其中MA是8x6,每个item负责一行
        for(int col = 0; col < 6; col++){
            float data = MA[lid][col];
            for (int row = 0; row < 6; row++){
                temp_row[lid] = AT[row][lid] * data;
                barrier(CLK_LOCAL_MEM_FENCE);
                // 累加得到该元素值
                if(lid == 0){
                    output[i*36 + row*6 + col] += temp_row[0]+temp_row[1]+temp_row[2]+temp_row[3]+
                                                    temp_row[4]+temp_row[5]+temp_row[6]+temp_row[7];
                }
            }

        }
    }

}

__kernel void tiles2img( __global float *Y,
                         __global float *output,
                         int inc,
                         int outc,
                         int tiles_w,
                         int tiles_h
){
    unsigned int idx = get_global_id(0);
    unsigned int total_items = get_global_size(0);
    unsigned int size_per_map = 36 * tiles_w * tiles_h; // 一个特征图（完整的）所拥有的像素数量
    for(int i = idx; i < 36 * outc * tiles_w * tiles_h; i+=total_items){
        // 得到对应的输出通道
        unsigned int n = i / size_per_map;
        // 得到对应的行和列
        unsigned int row = (i % size_per_map) / (6 * tiles_w);
        unsigned int col = (i % size_per_map) % (6 * tiles_w);
        // 判断属于这个输入通道的哪一个tile
        unsigned int tile_w = col / 6;
        unsigned int tile_h = row / 6;
        unsigned int t = tile_w + tile_h * tiles_w;
        // 判断在tile(6x6)中的相对位置
        unsigned int row_t = row % 6;
        unsigned int col_t = col % 6;
        unsigned int rel_pos = row_t*6 + col_t;
        // 找到Y中对应输出通道的索引
        unsigned int data = n * inc * size_per_map;
//        printf("%f",i);
        // 遍历每一个输入通道求和(此时tiles还没有合并)
        float sum=0.0f;
        for(int c = 0; c < inc; c++){
            sum += Y[data + c * size_per_map + t*36 + rel_pos];

        }
        output[i] = sum;

    }
}