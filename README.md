## 目录
- [简介](#简介)
- [测试](#测试)

## 简介
- **OpenCL**  
- **采用HWC内存布局**  
- **Real-ESRGAN超分模型**
```plaintext
|── core/
│   ├── include                 所有头文件
│   ├── cl_context              OpenCL的配置以及一些全局设置
│   ├── memory                  
│         ├── manager.cpp       所有申请的内存都在此进行管理与释放
│         └── tensor.cpp        输入以及输出均用tensor表示，默认是In-place的
│   ├─── model  
│         ├── loader.cpp        模型权重加载器
│         └── real_esrgan.cpp   定义模型结构
│   ├─── op                     
│         ├── common            基础算子
│         ├── conv              卷积算子
│         └── upsampler         上采样算子
│   └── CMakeLists.txt     
```

## 测试

