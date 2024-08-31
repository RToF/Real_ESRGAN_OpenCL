//
// Created by scz on 24-2-23.
//
#include <opencv2/core.hpp>     // opencv.hpp太重了
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "real_esrgan.h"

int main() {
    // --------------------------- 加载权重 ---------------------------------//
    Loader loader("../real_esrgan_param.bin");
    real_esrgan model(32, 3, 64);
    model.load(loader);
    // --------------------------- 读取图片 ---------------------------------//
    cv::Mat img = cv::imread("/home/scz/Pictures/2.jpeg", cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255);

    auto input_data = reinterpret_cast<INPUT_TYPE *>(img.data);
    tensor x(static_cast<short>(img.rows), static_cast<short>(img.cols), static_cast<short>(img.channels()),
             input_data);

    // 打印图片信息
    std::cout << "Image binary depth: " << img.depth() << std::endl;
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "Image channels: " << img.channels() << std::endl;
    // --------------------------- 进行超分 ---------------------------------//
    auto start = std::chrono::high_resolution_clock::now();

    model.forward(x);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "操作执行时间: " << elapsed.count() << " 秒" << std::endl;
    // --------------------------- 保存图片 ---------------------------------//
    size_t buffers_size = x.w() * x.h() * x.c() * sizeof(INPUT_TYPE);

    // 分配主机内存
    auto *hostData = (unsigned char *) malloc(buffers_size);

    // 从 OpenCL 缓冲区读取数据
    clEnqueueReadBuffer(ctx::config.commandQueue, x.data(), CL_TRUE, 0, buffers_size, hostData, 0, NULL, NULL);
    cv::Mat out(x.h(), x.w(), CV_32FC3, hostData);
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_8UC3, 255);

    if (!cv::imwrite("../1_cl.jpeg", out)) {
        std::cerr << "图像保存失败!" << std::endl;
        return -1;
    }

    std::cout << "图像保存成功!" << std::endl;
    return 0;
}

/*
    // ---------------------------- 正常显示图片 --------------------------------------------//
    auto input_data = reinterpret_cast<INPUT_TYPE*>(img.data);
    tensor x(static_cast<short>(img.rows), static_cast<short>(img.cols), static_cast<short>(img.channels()),input_data);

    cl_int err;
    size_t buffers_size = x.w()*x.h()*x.c()*sizeof(INPUT_TYPE);

    // 分配主机内存
    unsigned char *hostData = (unsigned char *)malloc(buffers_size);

    // 从 OpenCL 缓冲区读取数据
    clEnqueueReadBuffer(ctx::config.commandQueue, x.data(), CL_TRUE, 0, buffers_size, hostData, 0, NULL, NULL);
    cv::Mat out(x.h(), x.w(), CV_32FC3, hostData);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_8UC3, 255);

    // 显示图像
    cv::imshow("Image", out);
    cv::waitKey(0);
    return 0;
    // ---------------------------- 正常显示图片 --------------------------------------------//

    // --------------------------------- //
    using json = nlohmann::json;
    json j;
    j["data"] = data;
    std::ofstream file("data.json");
    if (file.is_open()) {
        file << j.dump(4);  // dump(4) 用于美化输出
        file.close();
    } else {
        std::cerr << "无法打开文件!" << std::endl;
    }
    // --------------------------------- //

    // 将 tensor 转换为 cv::Mat
    cv::Mat output_img = tensorToMat(x);

    // 保存处理后的图像
    if (!cv::imwrite("../1_cl.jpeg", output_img)) {
        std::cerr << "图像保存失败!" << std::endl;
        return -1;
    }

    std::cout << "图像保存成功!" << std::endl;
    return 0;

    // 读取模型参数
    struct Loader loader1("../conv1_weights_and_bias.bin");
    struct Loader loader2("../conv2_weights_and_bias.bin");
    layer::Conv2d<MODEL_TYPE> conv1(3,2,5,2, true);
    layer::Conv2d_prelu<MODEL_TYPE> conv2(2,2,3,2, true);
    conv1.load(loader1);
    conv2.load(loader2);

    if (check_fp16_support(ctx::config.deviceId))
        std::cout << "Device supports cl_khr_fp16." << std::endl;

    // --------------------------------------------------------------------------------//
    // TODO:使用cl_image
    // 读取图片

    cv::Mat img = cv::imread("/home/scz/Pictures/4.png", cv::IMREAD_COLOR);

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255);

    for (int i=0;i<10;i++){
        std::cout << reinterpret_cast<INPUT_TYPE*>(img.data)[i] << "\t";
    }

    // 转换颜色空间为 RGB
    auto input_data = reinterpret_cast<INPUT_TYPE*>(img.data);
    tensor x(static_cast<short>(img.rows), static_cast<short>(img.cols), static_cast<short>(img.channels()),input_data);

    // 打印图片信息
    std::cout << "Image binary depth: " << img.depth() << std::endl;
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "Image channels: " << img.channels() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //运算p
    conv1.forward(x);
    conv2.forward(x);

    // 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "操作执行时间: " << elapsed.count() << " 秒" << std::endl;

    return 0;

}

// --------------------------------------------------------------------------------//
//    用opencv作为输入
    short h = 5, w=8, c=3;
    int length = h*w*c;  // 可以根据需要修改长度上限

    // 创建并填充 vector
    std::vector<float> data(length);
    for (int i = 0; i < length; ++i) {
        data[i] = static_cast<float>(i);  // 递增的浮点数
    }

    // 输出数据以验证
    for (float value : data) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    cv::Mat img(h, w, CV_32FC3,data.data());
    auto input_data = reinterpret_cast<INPUT_TYPE*>(img.data);
    tensor x(h, w, c,input_data);
// --------------------------------------------------------------------------------//

//    用std::vector直接作为输入
    auto x = std::vector<float>(3*4*8,1.f);
    manager::buffer* input_buffer;
    size_t input_bs = x.size()  * sizeof(INPUT_TYPE);

    input_buffer = manager::allocator.get(input_bs, &input_buffer);
    err = clEnqueueWriteBuffer(ctx::config.commandQueue, input_buffer->ptr, CL_NON_BLOCKING,
                               0, input_buffer->bs, x.data(), 0, nullptr,
                               nullptr);
    CHECK(err, "create input buffer failed: ");
// --------------------------------------------------------------------------------//
*/