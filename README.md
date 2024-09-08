## Table of Contents
- [Introduction](#introduction)
- [Usage Steps](#usage-steps)
- [Model Setup](#model-setup)
- [Notes](#notes)
- [TODO](#todo)

## Introduction
- **OpenCL Inference Framework**  
- **HWC Format Operators**  
- **Real-ESRGAN Super-Resolution Model**
```plaintext
|── core/
|   ├── include                 All header files
|   ├── cl_context              OpenCL configuration and global settings
|   ├── memory                  
|   │   ├── manager.cpp         Memory management and release
|   │   └── tensor.cpp          Input and output are represented as tensors (default In-place)
|   ├── model  
|   │   ├── loader.cpp          Model weight loader
|   │   └── real_esrgan.cpp     Model structure definition
|   └── op                     
|       ├── common              Basic operators
|       ├── conv                Convolution operators
|       └── upsampler           Upsampling operators
|    
|── real_esrgan.py              PyTorch version (for validation and comparison)
|── demo.cpp                    Example usage
|── real_esrgan_param.bin       Model binary weights
|── SRVGGNetCompact.pth         PyTorch model weights
```
## Usage Steps
1. Load weights into the model
```c++
    real_esrgan model(32, 3, 64);                   // Create model
    Loader loader("../real_esrgan_param.bin");      // Create loader
    model.load(loader);                             // Load weights
```
2. Prepare input
```c++
    tensor x(height, width, channel, input_data);    // input_data is a pointer to the data
```

3. Perform inference
```c++
    model.forward(x);
```

4. Read results to Host
```c++
    size_t buffers_size = x.w() * x.h() * x.c() * sizeof(INPUT_TYPE);

    // Allocate host memory
    auto *hostData = (unsigned char *) malloc(buffers_size);

    // Read data from OpenCL buffer
    clEnqueueReadBuffer(ctx::config.commandQueue, x.data(), CL_TRUE, 0, buffers_size, hostData, 0, NULL, NULL);
```
## Model Setup
```c++
class real_esrgan {
private:
    std::vector<layer::Conv2d_prelu<MODEL_TYPE>> body;
    layer::Conv2d<MODEL_TYPE> tail;
    layer::PixelShuffle<INPUT_TYPE> upsampler1;
    layer::Interpolate<INPUT_TYPE> upsampler2;
public:
    real_esrgan(unsigned char body_num, short c, short mid_c, short ks=3, short _scale = 4);

    void load(Loader& loader){          // Load weights sequentially

        for (auto& layer: body){
            layer.load(loader);
        }

        tail.load(loader);
    }

    void forward(tensor& input){        // Perform inference sequentially
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

## Notes
- If the system hangs during execution, adjust the indexSpaceSize in core/include/conv to 1024 for RK3568 to run properly.
  
## TODO
- [ ] Winograd convolution acceleration
- [ ] DFS for computation graph construction
- [ ] Concurrent scheduling of operators
- [ ] Write unit tests with GTest
- [ ] Add ONNX parsing and exporting


