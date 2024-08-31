//
// Created by scz on 24-2-26.
//

#ifndef CONVERT_TENSOR_H
#define CONVERT_TENSOR_H
#include "manager.h"
#include <iomanip>

class tensor {
private:
    manager::buffer *buf;
    short height;
    short width;
    short channel;
    static cl_kernel add_op;
public:
    tensor()=delete;
    tensor(short _height, short _width, short _channel, INPUT_TYPE* _data);
    tensor(const tensor& other);
    ~tensor();

    void add(tensor& other);

    void make_buf_available();

    static void set_op(cl_kernel op);

    short w() const;
    void set_w (short w);

    short h() const;
    void set_h (short h);

    short c() const;
    void set_c (short c);

    cl_mem& data() const;
    void set_data(manager::buffer* result);

};


#endif //CONVERT_TENSOR_H
