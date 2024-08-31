//
// Created by scz on 24-2-23.
//
#include <opencv2/core.hpp>     // opencv.hpp太重了
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "real_esrgan.h"

int main() {
    // --------------------------- 加载权重 ---------------------------------//
    Loader loader("../weights/real_esrgan_param.bin");
    real_esrgan model(32, 3, 64);
    model.load(loader);
    // --------------------------- 读取图片 ---------------------------------//
    cv::Mat img = cv::imread("Your img path", cv::IMREAD_COLOR);
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

    if (!cv::imwrite("Save path", out)) {
        std::cerr << "图像保存失败!" << std::endl;
        return -1;
    }

    std::cout << "图像保存成功!" << std::endl;
    return 0;
}
