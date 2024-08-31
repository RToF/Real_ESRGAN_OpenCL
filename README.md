## 目录
- [简介](#简介)
- [使用步骤](#使用步骤)
- [测试](#测试)

## 简介
- **OpenCL**  
- **HWC内存布局**  
- **Real-ESRGAN超分模型**
```plaintext
|── core/
    ├── include                 所有头文件
    ├── cl_context              OpenCL的配置以及一些全局设置
    ├── memory                  
    │   ├── manager.cpp         所有申请的内存都在此进行管理与释放
    │   └── tensor.cpp          输入以及输出均用tensor表示，默认是In-place的
    ├── model  
    │   ├── loader.cpp          模型权重加载器
    │   └── real_esrgan.cpp     定义模型结构
    ├── op                     
    │   ├── common              基础算子
    │   ├── conv                卷积算子
    │   └── upsampler           上采样算子
    └── CMakeLists.txt     
```
## 使用步骤
1. 将权重加载到模型中
```c++
    real_esrgan model(32, 3, 64);                   // 建立模型
    Loader loader("../real_esrgan_param.bin");      // 建立读取器
    model.load(loader);                             // 读入权重
```
2. 创建输入tensor
```c++
    tensor x(height, width, channel, input_data);    // input_data为指向数据的指针
```

3. 进行运算
```c++
    model.forward(x);
```

4. 对结果进行的操作（以保存图片为例）
```c++
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

```
## 测试

