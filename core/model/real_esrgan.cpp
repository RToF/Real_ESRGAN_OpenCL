//
// Created by scz on 24-2-28.
//

#include "real_esrgan.h"


real_esrgan::real_esrgan(unsigned char body_num, short c, short mid_c, short ks, short _scale)
        : body(), tail(mid_c, c * _scale * _scale, ks), upsampler1(_scale, c * _scale * _scale),
          upsampler2(_scale){
    body.reserve(body_num+1);

    // head
    body.emplace_back(c, mid_c, ks);

    // body
    for (int i = 0; i < body_num; i++){
        body.emplace_back(mid_c, mid_c , ks);
    }

}

void real_esrgan::load(Loader& loader){
    // 加载body和head部分的参数
    for (auto& layer: body){
        layer.load(loader);
    }
    // 加载尾部的参数
    tail.load(loader);
}

void real_esrgan::forward(tensor& input){
    tensor base = input;

    // body
    for (auto& layer: body){
        layer.forward(input);
    }

    // tail
    tail.forward(input);
    upsampler1.forward(input);
    upsampler2.forward(base);
    input.add(base);
}