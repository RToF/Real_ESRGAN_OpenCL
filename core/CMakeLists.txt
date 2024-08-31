

add_library(core SHARED
        cl_context/cl_context.cpp
        memory/manager.cpp
        model/real_esrgan.cpp
        memory/tensor.cpp
        model/loader.cpp
)

target_include_directories(core PUBLIC ${OpenCL_INCLUDE_DIRS})
target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC  ${OpenCL_LIBRARIES})
target_link_libraries(core PUBLIC ${OpenCV_LIBS} )

# 是否打印 DEBUG 信息
#target_compile_definitions(core PUBLIC CONV_DEBUG)
#target_compile_definitions(core PUBLIC MANAGER_DEBUG)