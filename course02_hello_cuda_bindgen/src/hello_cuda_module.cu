extern "C" {

// CUDA核函数（CPU主机端调用，GPU设备端执行）
// CUDA核函数标识符：__global__
// CUDA核函数返回值类型必须是void
__global__ void hello_cuda_from_gpu(int n) {
    printf("GPU: 你好, CUDA! (Rust版)\n");
}

}