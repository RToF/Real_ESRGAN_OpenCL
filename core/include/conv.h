//
// Created by scz on 24-2-24.
//

#ifndef CONVERT_CONV_H
#define CONVERT_CONV_H

#include <vector>
#include "loader.h"
#include "tensor.h"

namespace layer{

    template <typename T>
    class baseLayer{
    private:
        mutable manager::buffer* output_buffer; // 由内存管理器统一释放
        cl_kernel op;                       // 指针，指向区域统一由cl_context释放
        const short in_c;
    public:
        explicit baseLayer(short _in_c):output_buffer(nullptr), op(nullptr), in_c(_in_c){}
        ~baseLayer()=default;

        virtual void forward(tensor& input){};

        short inc(){
            return in_c;
        }

        void set_op(cl_kernel kernel){
            op = kernel;
        }

        cl_kernel get_op() const {
            return op;
        }

        manager::buffer* &get_out_buf() const {
            return output_buffer;
        }

    };

    template <typename T>
    class Conv2d:public baseLayer<T>{
    private:
        const bool with_bias;                   // 无pad，默认按标准pad
        const int8_t stride;                    // int_fast8_t
        const int8_t kernel_size;

        const short out_c;
        manager::buffer* w_buf;                 // weight的buffer,若有bias则也放进去
        size_t total_bs;


    public:
        Conv2d()=delete;
        Conv2d(short _in_c, short _out_c, int8_t _ks, int8_t _stride=1, bool _with_bias= true)
                :out_c(_out_c), kernel_size(_ks), stride(_stride), with_bias(_with_bias),
                total_bs(this->inc() * out_c * kernel_size * kernel_size  * sizeof(T) + out_c * sizeof(T) * with_bias),
                w_buf(nullptr),baseLayer<T>(_in_c){
            this->set_op(ctx::config.ops_dict["Conv2d"]);
        };
        ~Conv2d()=default;

        virtual void load(struct Loader& loader);

        void forward(tensor& input) override;


        short outc(){
            return out_c;
        }
        short ks(){
            return kernel_size;
        }

        manager::buffer* &buf(){ // 返回指针的引用
            return w_buf;
        }

        bool bias(){
            return with_bias;
        };

        size_t bs() const{
            return total_bs;
        }

        void add_bs(size_t offset){
            total_bs += offset;
        }


    };

    template <typename T>
    class Conv2d_prelu: public Conv2d<T>{
    public:
        Conv2d_prelu(short _in_c, short _out_c, int8_t _ks, int8_t _stride=1, bool _with_bias= true)
                :Conv2d<T>(_in_c, _out_c, _ks, _stride, _with_bias) {
            this->set_op(ctx::config.ops_dict["Conv2d_prelu"]);
            this->add_bs(sizeof(T) * _out_c); // 加上 PRelu的参数
        }

        void load(Loader& loader) override{
            this->buf() = manager::allocator.get(this->bs(), &this->buf());
            cl_int err;
            // weight
            size_t weight_bs = this->inc() * this->outc() * this->ks() * this->ks()  * sizeof(T);
            void *ptr = reinterpret_cast<char*>(loader.mapped_file) + loader.offset;
            err = clEnqueueWriteBuffer(ctx::config.commandQueue, this->buf()->ptr, CL_NON_BLOCKING, 0, weight_bs, ptr, 0, nullptr,
                                       nullptr);
            // TODO:报错信息根据类名来
            CHECK(err, "Write buffer failed in Conv2d_prelu weight: ");
            loader.offset += weight_bs;

            if(this->bias()){
                size_t bias_bs = this->outc() * sizeof(T);
                ptr = reinterpret_cast<char*>(loader.mapped_file) + loader.offset;
                err = clEnqueueWriteBuffer(ctx::config.commandQueue, this->buf()->ptr, CL_NON_BLOCKING, weight_bs, bias_bs, ptr, 0, nullptr,
                                           nullptr);
                CHECK(err, "Write buffer failed in Conv2d_prelu bias: ");
                loader.offset += bias_bs;
            }

            // PRelu
            ptr = reinterpret_cast<char*>(loader.mapped_file) + loader.offset;
            err = clEnqueueWriteBuffer(ctx::config.commandQueue, this->buf()->ptr, CL_NON_BLOCKING, weight_bs + this->bias() * this->outc() * sizeof(T), sizeof(T) * this->outc(), ptr, 0, nullptr,
                                       nullptr);
            CHECK(err, "Write buffer failed in Conv2d_prelu prelu: ");
            loader.offset += sizeof(T) * this->outc();
        }
    };

/*
    template <typename T>
    class Conv2d_leaky_relu :public Conv2d<T> {
    public:
        Conv2d_leaky_relu(short _in_c, short _out_c, int8_t _ks, int8_t _stride=1, bool _with_bias= true)
                : Conv2d<T>(_in_c, _out_c, _ks, _stride, _with_bias) {
            this->set_op(ctx::config.ops_dict["Conv2d_leaky_relu"]);
        }
    };
*/

    // -------------------------------- 具体实现 ------------------------------------ //
    template <typename T>
    void Conv2d<T>::load(Loader& loader){
        w_buf = manager::allocator.get(total_bs, &w_buf);
        cl_int err;
        // weight
        size_t weight_bs = this->inc() * out_c * kernel_size * kernel_size  * sizeof(T);
        clFinish(ctx::config.commandQueue);
        void *ptr = reinterpret_cast<char*>(loader.mapped_file) + loader.offset;
        // TODO:需要加上等待全部分配完成的指令
        err = clEnqueueWriteBuffer(ctx::config.commandQueue, w_buf->ptr, CL_NON_BLOCKING, 0, weight_bs, ptr, 0, nullptr,
                                   nullptr);
        CHECK(err, "Write buffer failed in Conv2d weight: ");
        loader.offset += weight_bs;

        if(with_bias){
            size_t bias_bs = out_c * sizeof(T);
            ptr = reinterpret_cast<char*>(loader.mapped_file) + loader.offset;
            err = clEnqueueWriteBuffer(ctx::config.commandQueue, w_buf->ptr, CL_NON_BLOCKING, weight_bs, bias_bs, ptr, 0, nullptr,
                                       nullptr);
            CHECK(err, "Write buffer failed in Conv2d bias: ");
            loader.offset += bias_bs;
        }
    }

    template <typename T>
    void Conv2d<T>::forward(tensor& input){
        auto inc = this->inc();
        if (input.c() != inc){
            std::cerr << "input.c != conv.c" << std::endl;
        }
        // 输出
        short in_h, in_w, out_h, out_w;
        auto padding = (kernel_size - 1) / 2;
        in_h = input.h();
        in_w = input.w();
        out_h = (in_h - kernel_size + 2 * padding) / stride + 1;
        out_w = (in_w - kernel_size + 2 * padding) / stride + 1;

        unsigned int out_size = out_h * out_w * out_c;
        this->get_out_buf() = manager::allocator.get(out_size * sizeof(T), &this->get_out_buf());

        // 设置参数
        SET_ARGS(0, input.data());
        SET_ARGS(1, w_buf->ptr);
        SET_ARGS(2, this->get_out_buf()->ptr);
        SET_ARGS(3, out_size);
        SET_ARGS(4, kernel_size);
        SET_ARGS(5, stride);
        SET_ARGS(6, with_bias);
        SET_ARGS(7, in_h);
        SET_ARGS(8, in_w);
        SET_ARGS(9, inc);
        SET_ARGS(10, out_h);
        SET_ARGS(11, out_w);
        SET_ARGS(12, out_c);

        // TODO:设置防止超过上限
        // 设置work_item和work_group
        size_t indexSpaceSize[1] = {out_size};//所有group的item
        size_t workGroupSize[1] = {static_cast<size_t>(out_c)};//每个group的item
//        if (out_size )

        clFinish(ctx::config.commandQueue); // 等待前述操作完成（前面的forward/load）
        //开始运行
        clEnqueueNDRangeKernel(ctx::config.commandQueue, this->get_op(), 1, nullptr, indexSpaceSize, workGroupSize, 0, nullptr,
                               nullptr);

        //修改tensor
        input.set_h(out_h);
        input.set_w(out_w);
        input.set_c(out_c);
        input.set_data(this->get_out_buf());

#ifdef CONV_DEBUG
        // 打印输出
        auto out = new float[out_size];
        std::fill(out, out + out_size, 0.0f);
        clFinish(ctx::config.commandQueue);
        clEnqueueReadBuffer(ctx::config.commandQueue, this->get_out_buf()->ptr, CL_TRUE, 0, out_size * sizeof(INPUT_TYPE), out, 0, nullptr, nullptr);
        std::cout << "------------------------------------ type:HWC --------------------------------------"<<std::endl;
        out_size = out_size > 20 ? 20:out_size;
        for (size_t i = 0; i < out_size; i++) {
            std::cout << std::setw(12) <<out[i] << "\t";
            if ((i + 1) % 4 == 0) {
                std::cout << std::endl;
            }
        }
        delete[] out;
#endif
    }




}

#endif //CONVERT_CONV_H
