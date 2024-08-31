//
// Created by scz on 24-2-31.
//
#include "loader.h"

Loader::Loader(const char *_file_name): file_name(_file_name), offset(0){
    fd = open(file_name, O_RDONLY);
    if (fd == -1) {
        std::cerr << "open error: " << strerror(errno) << std::endl;
        return;
    }

    file_size = lseek(fd, 0, SEEK_END);
    if (file_size == -1) {
        std::cerr << "lseek error: " << strerror(errno) << std::endl;
        close(fd);
        return;
    }
    lseek(fd, 0, SEEK_SET); // 重置文件偏移量

    // 映射文件到内存
    mapped_file = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_file == MAP_FAILED) {
        std::cerr << "mmap error: " << strerror(errno) << std::endl;
        close(fd);
        return;
    }
}

Loader::~Loader(){
    if (munmap(mapped_file, file_size) == -1) {
        std::cerr << "munmap error: " << strerror(errno) << std::endl;
    }
    close(fd);
}