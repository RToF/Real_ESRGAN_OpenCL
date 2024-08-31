//
// Created by scz on 24-2-26.
//

#include "manager.h"

namespace manager{

#ifdef MANAGER_DEBUG
    int buffer::release_num = 1;
    int Manager::create_num = 1;
    unsigned int Manager::max_mem = 0;
    unsigned int Manager::cur_mem = 0;
#endif

    Manager allocator; //全局内存分配器

    buffer::buffer(size_t _size, buffer** _owner,bool _used): used(_used), bs(_size), owner(_owner){  //没有host数据的情况,先创建再往里面写
        cl_int err;
        ptr = clCreateBuffer(ctx::config.context, CL_MEM_READ_ONLY, _size,
                             nullptr, &err);
        CHECK(err, "Error creating buffer in buffer(): ");
    }
    buffer::~buffer(){
#ifdef MANAGER_DEBUG
        std::cout << "release " << release_num++ << "th buffer in "<< owner << std::endl;
#endif
        if(ptr){
            clReleaseMemObject(ptr);
        }
        ptr = nullptr;
        owner = nullptr;
#ifdef MANAGER_DEBUG
        Manager::cur_mem -= bs;
#endif
    }
        //移动构造
    buffer::buffer(buffer&& other) noexcept
            : used(other.used), ptr(other.ptr), bs(other.bs) {
        bs = other.bs;
        ptr = other.ptr;
        used = other.used;
        owner = other.owner;
        *owner = this;          // 指向新的buffer
        other.ptr = nullptr;    // 移动后，原对象不再持有资源
        other.bs = 0;
        other.used = false;
        other.owner = nullptr;
    }

    void buffer::deepcopy(const buffer* other, buffer** _owner){
        // 只能对正在使用的进行拷贝（也可以换成判断bs是否为0之类的）
        if (other->used){
            cl_int err;

            err = clEnqueueCopyBuffer(ctx::config.commandQueue, other->ptr, ptr,
                                      0, 0, other->bs,0,
                                      nullptr, nullptr);
            CHECK(err, "copy buffer: ");

            used = true;
            bs = other->bs;
            owner = _owner;
        }
    }


    Manager::Manager(){storage.reserve(RESERVED_BUFFER_NUM);}  // 先预留空间

    Manager::~Manager() {
#ifdef MANAGER_DEBUG
        std::cout << "max_mem: " << max_mem << std::endl;
#endif
    };

    buffer* Manager::get(size_t size, buffer** owner){
        // 先从storage里找
        buffer *back;
        size_t gap = size; // 表示buffer和想要的空间的差距（大于0） 初始化为size(可以避免选到大于2*size的)
        // 开始查找
        for(auto& buf:storage){
            if (buf.used) continue;

            // 计算当前 buf 的差值
            auto temp = buf.bs >= size ? buf.bs - size : gap;

            // 更新 gap 为最小差值
            if (temp < gap) {
                back = &buf;
                gap = temp;
            }
        }
        // TODO： 测试
        // 找到的情况, 即gap变动了
        if (gap < size){
            // 当size小于50MB
            if (back->bs < 50 * 1024 * 1024){
                back->used = true;
                back->owner = owner;
                //back->bs = size; //size不能改变，因为这是已经存在的存储空间，大小固定
                return back;
            }
            // 当size大于50MB  &&  比size大的不多
            if (back->bs > 50 * 1024 * 1024 && back->bs * 0.8 < size){
                back->used = true;
                back->owner = owner;
                //back->bs = size;
                return back;
            }
        }

        /*
         *   1.storage扩容时候会拷贝opencl分配的buffer，需要实现移动/深拷贝
         *   2.因此也不能用push_back，会拷贝副本
         *       这里选择实现buffer的移动语义
         */

        // 没找到或是不满足就直接分配
        storage.emplace_back(size, owner);
#ifdef MANAGER_DEBUG
        cur_mem += size;
        if (max_mem < cur_mem){
            max_mem = cur_mem;
        }

        std::cout << "create " << create_num++ << "th buffer in " << owner << std::endl;
#endif
        return &storage.back();
    };
}
