//
// Created by scz on 24-2-28.
//

#ifndef CONVERT_REAL_ESRGAN_H
#define CONVERT_REAL_ESRGAN_H
#include "conv.h"
#include "upsampler.h"

class real_esrgan {
private:
    std::vector<layer::Conv2d_prelu<MODEL_TYPE>> body;
    layer::Conv2d<MODEL_TYPE> tail;
    layer::PixelShuffle<INPUT_TYPE> upsampler1;
    layer::Interpolate<INPUT_TYPE> upsampler2;

public:
    real_esrgan(unsigned char body_num, short c, short mid_c, short ks=3, short _scale = 4);

    void load(Loader& loader);

    void forward(tensor& input);
};


#endif //CONVERT_REAL_ESRGAN_H
