//
// Created by scz on 24-2-30.
//
#include "tensor.h"

cl_kernel tensor::add_op;

tensor::~tensor(){
    make_buf_available();
};

tensor::tensor(const tensor& other){
    if(this == &other){
        std::cerr << "tensor can not equal to itself" << std::endl;
    }
    buf = manager::allocator.get(other.buf->bs, &buf);
    this->buf->deepcopy(other.buf, &buf);
    height = other.h();
    width = other.w();
    channel = other.c();
}

void tensor::add(tensor& other){
    cl_int err;
    unsigned int out_size = height * width * channel;

    err = clSetKernelArg(tensor::add_op, 0, sizeof(this->data())  , (void *) &this->data());
    CHECK(err, "fail in tensor.add() setKernelArg1");
    err = clSetKernelArg(tensor::add_op, 1, sizeof(other.data())  , (void *) &other.data());
    CHECK(err, "fail in tensor.add() setKernelArg2");
    err = clSetKernelArg(tensor::add_op, 2, sizeof(out_size)      , (void *) &out_size    );
    CHECK(err, "fail in tensor.add() setKernelArg3");

    size_t indexSpaceSize[1] =  {out_size};                      //所有group的item
    size_t workGroupSize[1]  =  {static_cast<size_t>(channel)};   //每个group的item

    clEnqueueNDRangeKernel(ctx::config.commandQueue, add_op, 1, nullptr, indexSpaceSize, workGroupSize, 0, nullptr,
                           nullptr);

#ifdef CONV_DEBUG
    // 打印输出
    auto out = new float[out_size];
    std::fill(out, out + out_size, 0.0f);
    clFinish(ctx::config.commandQueue);
    clEnqueueReadBuffer(ctx::config.commandQueue, this->buf->ptr, CL_TRUE, 0, out_size * sizeof(INPUT_TYPE), out, 0, nullptr, nullptr);
    std::cout << "------------------------------------ type:HWC --------------------------------------"<<std::endl;
    out_size = out_size > 20 ? 20:out_size;
    for (size_t i = 0; i < out_size/2; i++) {
        std::cout << std::setw(12) <<out[i] << "\t";
        if ((i + 1) % 4 == 0) {
            std::cout << std::endl;
        }
    }
    delete[] out;
#endif
}

void tensor::make_buf_available(){ // 让之前的buf变为可用状态
    buf->used = false;
    buf->owner = nullptr;
}

void tensor::set_op(cl_kernel op){
    add_op = op;
}

tensor::tensor(short _height, short _width, short _channel,INPUT_TYPE* _data):height(_height), width(_width), channel(_channel){
    cl_int err;
    size_t tensor_bs = height * width * channel * sizeof(INPUT_TYPE);
    buf = manager::allocator.get(tensor_bs, &buf);
    err = clEnqueueWriteBuffer(ctx::config.commandQueue, buf->ptr, CL_NON_BLOCKING,
                               0, buf->bs, _data, 0, nullptr,
                               nullptr);
    CHECK(err, "create tensor failed: ");
}

cl_mem& tensor::data() const{ //必须要返回引用，否则会复制(只实现移动，避免拷贝)
    return buf->ptr;
}

short tensor::w() const{
    return width;
}

void tensor::set_w (short w){
    width = w;
}

short tensor::h() const{
    return height;
}

void tensor::set_h (short h){
    height = h;
}

short tensor::c() const{
    return channel;
}

void tensor::set_c (short c){
    channel = c;
}

void tensor::set_data(manager::buffer* result){
    make_buf_available();
    buf = result;
}