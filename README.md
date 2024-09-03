## 目录
- [简介](#简介)
- [使用步骤](#使用步骤)
- [模型搭建](#模型搭建)
- [继承关系](#继承关系)
- [注意事项](#注意事项)

## 简介
- **OpenCL封装**  
- **HWC格式算子**  
- **Real-ESRGAN超分模型**
```plaintext
|── core/
|   ├── include                 所有头文件
|   ├── cl_context              OpenCL的配置以及一些全局设置
|   ├── memory                  
|   │   ├── manager.cpp         所有申请的内存都在此进行管理与释放
|   │   └── tensor.cpp          输入以及输出均用tensor表示(默认In-place)
|   ├── model  
|   │   ├── loader.cpp          模型权重加载器
|   │   └── real_esrgan.cpp     定义模型结构
|   └── op                     
|       ├── common              基础算子
|       ├── conv                卷积算子
|       └── upsampler           上采样算子     
|── real_esrgan.py              Pytorch版本（用于验证对比）
|── demo.cpp                    使用示例
|── real_esrgan_param.bin       模型的bin权重
|── SRVGGNetCompact.pth         模型的Pytorch权重
```
## 使用步骤
1. 将权重加载到模型中
```c++
    real_esrgan model(32, 3, 64);                   // 建立模型
    Loader loader("../real_esrgan_param.bin");      // 建立读取器
    model.load(loader);                             // 读入权重
```
2. 封装输入
```c++
    tensor x(height, width, channel, input_data);    // input_data为指向数据的指针
```

3. 进行运算
```c++
    model.forward(x);
```

4. 读取结果到Host端
```c++
    size_t buffers_size = x.w() * x.h() * x.c() * sizeof(INPUT_TYPE);

    // 分配主机内存
    auto *hostData = (unsigned char *) malloc(buffers_size);

    // 从 OpenCL 缓冲区读取数据
    clEnqueueReadBuffer(ctx::config.commandQueue, x.data(), CL_TRUE, 0, buffers_size, hostData, 0, NULL, NULL);
```
## 模型搭建
```c++
class real_esrgan {
private:
    std::vector<layer::Conv2d_prelu<MODEL_TYPE>> body;
    layer::Conv2d<MODEL_TYPE> tail;
    layer::PixelShuffle<INPUT_TYPE> upsampler1;
    layer::Interpolate<INPUT_TYPE> upsampler2;
public:
    real_esrgan(unsigned char body_num, short c, short mid_c, short ks=3, short _scale = 4);

    void load(Loader& loader){          // 依次加载权重

        for (auto& layer: body){
            layer.load(loader);
        }

        tail.load(loader);
    }

    void forward(tensor& input){        // 依次推理
        tensor base = input;
        
        for (auto& layer: body){
            layer.forward(input);
        }
    
        tail.forward(input);
        upsampler1.forward(input);
        upsampler2.forward(base);
        input.add(base);
    }
```
## 继承关系
```plaintext
|── BaseLayer
|   ├── Conv2d                  
|   │   ├── Conv2d_prelu 
|   │   └── Conv2d_leaky_relu         
|   └── UpSampler  
|       ├── PixelShuffle      
|       └── Interpolate
```
};
## 注意事项
- 若是在板端运行时卡住请调整core/include/conv中的indexSpaceSize大小，在RK3568设置为1024可以正常运行
