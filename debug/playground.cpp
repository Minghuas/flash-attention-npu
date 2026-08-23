#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;
using T = float;

constexpr int N = 16;
constexpr int SIZE = N * sizeof(T);

class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), N);
        g_y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), N);
        g_z.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z), N);
    }

    __aicore__ inline void Main()
    {
        Init();
        Load();
        Playground();
        Store();
    }

private:
    __aicore__ inline void Init()
    {
        // Allocate buffer space
        x = GetBuf<T>(0);
        y = GetBuf<T>(SIZE);
        z = GetBuf<T>(2 * SIZE);
    }

    __aicore__ inline void Load()
    {
        DataCopy(x, g_x, N);
        DataCopy(y, g_y, N);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void Playground()
    {
        // Do some computation
        Add(z, x, y, N);
    }

    __aicore__ inline void Store()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(g_z, z, N);
    }

private:
    template <typename U>
    __aicore__ inline LocalTensor<U> GetBuf(uint32_t offset)
    {
        return buf[offset].template ReinterpretCast<U>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> ub;
    LocalTensor<uint8_t> buf;

    LocalTensor<T> x, y, z;
    GlobalTensor<T> g_x, g_y, g_z;
};

__global__ __aicore__ void kernel_func(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel op(x, y, z);
    op.Main();
}

std::vector<T> launch_kernel(std::vector<T>& x, std::vector<T>& y)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_x = reinterpret_cast<uint8_t*>(x.data());
    uint8_t* h_y = reinterpret_cast<uint8_t*>(y.data());
    uint8_t* h_z = nullptr;
    uint8_t* d_x = nullptr;
    uint8_t* d_y = nullptr;
    uint8_t* d_z = nullptr;

    aclrtMallocHost(reinterpret_cast<void**>(&h_z), SIZE);
    aclrtMalloc(reinterpret_cast<void**>(&d_x), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_y), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_z), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_x, SIZE, h_x, SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_y, SIZE, h_y, SIZE, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<<<1, nullptr, stream>>>(d_x, d_y, d_z);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(h_z, SIZE, d_z, SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<T> z(reinterpret_cast<T*>(h_z), reinterpret_cast<T*>(h_z) + N);

    aclrtFreeHost(h_z);
    aclrtFree(d_x);
    aclrtFree(d_y);
    aclrtFree(d_z);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return z;
}

int main()
{
    std::vector<T> x(N, 1.0f);
    std::vector<T> y(N, 2.0f);
    std::vector<T> z = launch_kernel(x, y);
    for (const auto& val : z) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}
