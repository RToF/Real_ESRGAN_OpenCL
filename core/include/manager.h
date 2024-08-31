//
// Created by scz on 24-2-26.
//

#ifndef CONVERT_MANAGER_H
#define CONVERT_MANAGER_H

#define RESERVED_BUFFER_NUM 200

#include <vector>
#include "cl_context.h"

namespace manager{

    struct buffer{
        bool used;
        cl_mem ptr;
        size_t bs;              // byte_size, 不完全等于数据的空间，因为Manager可能会提供复用的buffer
        buffer** owner;         // 指向拥有者，当storage扩容的时候，同时需要更新拥有者指向的buffer
#ifdef MANAGER_DEBUG
        static int release_num; // 统计释放了多少个buffer
#endif
        buffer()=delete;
        buffer(size_t _size, buffer** _owner, bool _used = true);
        ~buffer();              // 统一在Manager里面的vector析构的时候释放

        buffer(buffer&& other) noexcept; //用于容器扩容时
        void deepcopy(const buffer* other, buffer** _owner);

    };

    class Manager {
    private:
        // TODO:把模型的buffer和其他的buffer分开，模型的一般不会被释放，加快检索速度
        std::vector<buffer> storage; // RAII自动析构释放里面的buffer
    public:
#ifdef MANAGER_DEBUG
        static int create_num;              // 统计创建了多少个buffer
        static unsigned int max_mem;        // 统计最大使用中内存开销
        static unsigned int cur_mem;        // 统计当前使用中内存开销
#endif
        Manager();  // 先预留空间
        ~Manager();

        buffer *get(size_t size, buffer **owner);

    };

    extern Manager allocator;

}



#endif //CONVERT_MANAGER_H
