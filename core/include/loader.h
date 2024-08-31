//
// Created by scz on 24-2-25.
//

#ifndef CONVERT_LOADER_H
#define CONVERT_LOADER_H

#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include "cl_context.h"

typedef struct Loader{
    const char *file_name;
    int fd;
    void *mapped_file; //mmap映射的内存
    off_t file_size;    //大小
    size_t offset; // 当前要读取位置的字节偏移量

    explicit Loader(const char *_file_name);
    ~Loader();

#ifdef CONV_DEBUG
    void show() const{
        for ( int i = 0; i < file_size/ sizeof(MODEL_TYPE); i++){
            std::cout << ((MODEL_TYPE*)mapped_file)[i] << " ";
            if ((i+1) % 10 == 0){
                std::cout << std::endl;
            }
        }
    }
#endif
}Loader;



#endif //CONVERT_LOADER_H
