//
// Created by scz on 24-2-26.
//
#include <iomanip>
#include "cl_context.h"
#include "tensor.h"

namespace ctx{
    cl_config config;

    cl_config::cl_config() : platformId(nullptr), deviceId(nullptr), context(nullptr), commandQueue(nullptr), program(nullptr){
        if (prepare_opencl() != CL_SUCCESS) {
            std::cerr << "Failed to initialize OpenCL." << std::endl;
        }
        get_ops();
    }

    cl_config::~cl_config() {
        if (program) clReleaseProgram(program);
        if (commandQueue) clReleaseCommandQueue(commandQueue);
        if (context) clReleaseContext(context);
        for (auto& entry : ops_dict) {
            if (entry.second) { // 检查内核是否有效
                clReleaseKernel(entry.second); // 释放内核对象
            }
        }
    }


    cl_int cl_config::prepare_opencl() {
        cl_int status;

        status = clGetPlatformIDs(1, &platformId, nullptr);
        if (status != CL_SUCCESS) return status;


        status = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
        if (status != CL_SUCCESS) return status;

        context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &status);

        // 查询并打印 GPU 型号
        char deviceName[100];
        status = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        if (status != CL_SUCCESS) return status;
        std::cout << "GPU Device Name: " << deviceName << std::endl;

        commandQueue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, &status);
        return status;
    }

    void cl_config::get_ops() {
        ops_dict["Conv2d"] = build_kernel("../core/op/conv/conv_hwc.cl", "Conv2d");
        ops_dict["Conv2d_prelu"] = build_kernel("../core/op/conv/conv_hwc_prelu.cl", "Conv2d_prelu");
        ops_dict["Conv2d_leaky_relu"] = build_kernel("../core/op/conv/conv_hwc_leaky_relu.cl", "Conv2d_leaky_relu");

        ////////////////////////////////////////// Winograd //////////////////////////////////////////
        ops_dict["GgGT"] = build_kernel("../core/op/conv/winograd63.cl", "GgGT");
        ops_dict["ATMA"] = build_kernel("../core/op/conv/winograd63.cl", "ATMA");
        ops_dict["UV"] = build_kernel("../core/op/conv/winograd63.cl", "UV");
        ops_dict["tiles2img"] = build_kernel("../core/op/conv/winograd63.cl", "tiles2img");
        ops_dict["BTdB"] = build_kernel("../core/op/conv/winograd63.cl", "BTdB");
        ops_dict["img2tiles"] = build_kernel("../core/op/conv/winograd63.cl", "img2tiles");
        ////////////////////////////////////////// Winograd //////////////////////////////////////////

        ops_dict["PixelShuffle"] = build_kernel("../core/op/upsampler/pixelshuffle_hwc.cl", "PixelShuffle");
        ops_dict["Interpolate"] = build_kernel("../core/op/upsampler/interpolate_hwc.cl", "Interpolate");

        ops_dict["Add"] = build_kernel("../core/op/common/add.cl", "Add");
        ops_dict["gemm"] = build_kernel("../core/op/common/gemm.cl", "gemm");
        ops_dict["img2col"] = build_kernel("../core/op/common/img2col_hwc.cl", "img2col");
        ops_dict["weight2col"] = build_kernel("../core/op/common/weight2col_hwc.cl", "weight2col");

        std::cout << "------------------------------------------ kernel info ----------------------------------------------" << std::endl;
        kernel_info(ops_dict["Conv2d"], "Conv2d");
        kernel_info(ops_dict["Conv2d_prelu"], "Conv2d_prelu");
        kernel_info(ops_dict["Conv2d_leaky_relu"], "Conv2d_leaky_relu");

        kernel_info(ops_dict["PixelShuffle"], "PixelShuffle");
        kernel_info(ops_dict["Interpolate"], "Interpolate");

        kernel_info(ops_dict["Add"], "Add");
        kernel_info(ops_dict["gemm"], "gemm");
        kernel_info(ops_dict["img2col"], "img2col");
        kernel_info(ops_dict["weight2col"], "weight2col");
        std::cout << "------------------------------------------ kernel info ----------------------------------------------" << std::endl;
        tensor::set_op(ctx::config.ops_dict["Add"]);
    }

    void cl_config::kernel_info(cl_kernel kernel, const std::string& message){
        std::cout << std::setw(20) << message << "\t|\t";
        //查看工作group和最大group
        size_t max_work_group_size;
        clGetDeviceInfo(ctx::config.deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size),
                        &max_work_group_size,
                        nullptr);
        std::cout << "Max Items Num per Group:" << max_work_group_size << " \t|\t";
        size_t kernel_max_group;
        clGetKernelWorkGroupInfo(kernel, ctx::config.deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_max_group),
                                 &kernel_max_group,
                                 nullptr);
        std::cout << "Max Groups in this Kernel:" << kernel_max_group << std::endl;
    }

    cl_kernel cl_config::build_kernel(const char *filename, const char *kernel_name) {
        std::ifstream kernel_file(filename, std::ios::in);
        if (!kernel_file.is_open()) {
            std::cerr << "Failed to open kernel file: " << filename << std::endl;
            return nullptr;
        }

        std::ostringstream oss;
        oss << kernel_file.rdbuf();
        std::string srcStdStr = oss.str();
        const char *srcStr = srcStdStr.c_str();
        cl_int status;

        program = clCreateProgramWithSource(context, 1, &srcStr, nullptr, &status);
        if (status != CL_SUCCESS) return nullptr;

        const char* options = "";
        status = clBuildProgram(program, 1, &deviceId, options, nullptr, nullptr);
        if (status != CL_SUCCESS) {
            // 打印错误
            size_t log_size;
            clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string build_log(log_size, '\0');
            clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], nullptr);
            std::cerr << "Build log:\n" << build_log << std::endl;
            clReleaseProgram(program);
            return nullptr;
        }

        cl_kernel kernel = clCreateKernel(program, kernel_name, &status);
        clReleaseProgram(program);
        if (status != CL_SUCCESS) return nullptr;

        return kernel;
    }
}
