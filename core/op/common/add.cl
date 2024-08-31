
__kernel void Add(  __global float* input1,
                    __global float* input2,
                    unsigned int out_size
){
    unsigned int idx = get_global_id(0);
    size_t global_items_sum = get_global_size(0);
    for (unsigned int i = idx; i < out_size; i += global_items_sum){
        input1[i] += input2[i];
    }

}