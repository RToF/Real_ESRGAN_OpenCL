//
// Created by scz on 24-2-24.
//

#ifndef CONVERT_CL_CONTEXT_H
#define CONVERT_CL_CONTEXT_H

#include <CL/cl.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

//#define RH_BUFFER(weight, size, err, result)                                                                            \
//                    cl_mem result = clCreateBuffer(ctx::config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size,   \
//                                    (void *) weight, &err);                                                             \
//                    CHECK(err, "Error creating buffer in RH_BUFFER: ")                                                  \

#define CHECK(err, message) do{                                                                                         \
    if (err != CL_SUCCESS) {                                                                                            \
        std::cerr << message << err << std::endl;}                                                                      \
        }while(0)                                                                                                       \

#define SET_ARGS(num, arg) clSetKernelArg(this->get_op(), num, sizeof(arg), (void *) &arg)

#define INPUT_TYPE float // 输入图像的类型
#define MODEL_TYPE INPUT_TYPE // 模型参数的类型


namespace ctx {

    struct cl_config {
        cl_platform_id platformId;
        cl_device_id deviceId;
        cl_context context;
        cl_command_queue commandQueue;
        cl_program  program;
        std::map<std::string, cl_kernel> ops_dict;

        cl_config();
        ~cl_config();

        cl_int prepare_opencl();
        void get_ops();
        static void kernel_info(cl_kernel kernel, const std::string& message);
        cl_kernel build_kernel(const char *filename, const char *kernel_name);
    };

    extern cl_config config; // 全局变量

/*
    //每个cl算子都要创建program和kernel
    typedef struct cl_func{
        cl_program program;
        cl_kernel kernel;
        cl_func():program(nullptr), kernel(nullptr), initialized(false) {};
        ~cl_func(){
            clReleaseKernel(kernel);
            clReleaseProgram(program);
        }
        bool initialized;

        // 用于编译指定的cl得到kernel
        void init_func(const char* filename, const char* kernel_name){
            std::ifstream kernel_file(filename, std::ios::in);
            std::ostringstream oss;
            oss << kernel_file.rdbuf();
            std::string srcStdStr = oss.str();
            const char *srcStr = srcStdStr.c_str();
            this->program = clCreateProgramWithSource(config.context, 1, &srcStr, nullptr, nullptr);
            clBuildProgram(this->program, 1, &config.deviceId, nullptr, nullptr, nullptr);
            this->kernel = clCreateKernel(this->program, kernel_name, nullptr);

            // TODO:（改成可选）获取编译日志
            size_t log_size;
            clGetProgramBuildInfo(program, config.deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char*)malloc(log_size);
            clGetProgramBuildInfo(program, config.deviceId, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("OpenCL Program Build Log:\n%s\n", log);
            free(log);
        }

        void initialize(const char *filename, const char *kernel_name = "conv2d"){
            if(!initialized){
                init_func(filename, kernel_name);
                initialized = true;
            }
        }

    }cl_func;
*/


}



#endif //CONVERT_CL_CONTEXT_H
