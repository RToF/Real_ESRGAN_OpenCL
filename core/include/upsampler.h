//
// Created by scz on 24-2-30.
//

#ifndef CONVERT_UPSAMPLER_H
#define CONVERT_UPSAMPLER_H
#include "conv.h"

namespace layer{

    template <typename T>
    class UpSampler: public baseLayer<T>{
    private:
        bool channel_chaned; // 是否会改变通道
        unsigned char scale;
    public:
        UpSampler()=delete;
        UpSampler(unsigned char _scale, short inc = 0): baseLayer<T>(inc), scale(_scale){}

        void set_channel_changed(bool c){
            channel_chaned = c;
        };

        void forward(tensor& input) override{
            short out_c, in_c;
            if (channel_chaned)
            {
                in_c = this->inc();
                if (input.c() != in_c){
                    // TODO:根据类名来提示报错信息
                    std::cerr << "input.c != in_c in Upsampler" << std::endl;
                }
                if (in_c % scale != 0){
                    std::cerr << "channels must be divisible by the scale in Upsampler" << std::endl;
                }
                out_c = in_c / scale / scale;
            }
            else{
                in_c = input.c();
                out_c = in_c;
            }

            short in_h, in_w, out_h, out_w;
            in_h = input.h();
            in_w = input.w();
            out_h = in_h * scale;
            out_w = in_w * scale;

            unsigned int out_size = out_h * out_w * out_c;
            this->get_out_buf() = manager::allocator.get(out_size * sizeof(T), &this->get_out_buf());

            // 设置参数
            SET_ARGS(0, input.data());
            SET_ARGS(1, this->get_out_buf()->ptr);
            SET_ARGS(2, out_size);
            SET_ARGS(3, in_h);
            SET_ARGS(4, in_w);
            SET_ARGS(5, in_c);
            SET_ARGS(6, out_h);
            SET_ARGS(7, out_w);
            SET_ARGS(8, out_c);
            SET_ARGS(9, scale);

            // TODO:使用local内存
//            auto local_size = in_c * scale * scale;  // 原图上的 scale * scale 大小, 每一个位置都有in_c个元素
//            clSetKernelArg(this->get_op(), 9, local_size * sizeof(T), NULL);

            // 设置work_item和work_group
            size_t indexSpaceSize[1] = {out_size};//所有group的item
            size_t workGroupSize[1] = {static_cast<size_t>(in_c)};//每个group的item

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
        std::cout << "------------------------------------ type:upsampler HWC --------------------------------------"<<std::endl;
        out_size = out_size > 40 ? 40:out_size;
        for (size_t i = 0; i < out_size; i++) {
            std::cout << std::setw(12) <<out[i] << "\t";
            if ((i + 1) % 3 == 0) {
                std::cout << std::endl;
            }
        }
        delete[] out;
#endif
        }
    };

    template <typename T>
    class PixelShuffle: public UpSampler<T>{
    public:
        PixelShuffle()=delete;
        PixelShuffle(unsigned char _scale, short _in_c): UpSampler<T>(_scale, _in_c){
            this->set_op(ctx::config.ops_dict["PixelShuffle"]);
            this->set_channel_changed(true);
        }
    };

    template <typename T>
    class Interpolate: public UpSampler<T>{
    public:
        Interpolate()=delete;
        Interpolate(unsigned char _scale): UpSampler<T>(_scale){
            this->set_op(ctx::config.ops_dict["Interpolate"]);
            this->set_channel_changed(false);
        }
    };

}

#endif //CONVERT_UPSAMPLER_H
