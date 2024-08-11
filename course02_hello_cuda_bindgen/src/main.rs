use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use anyhow::Result;

// bindgen_cuda直接复用hello_cuda_module.cu源文件
// bindgen_cuda将hello_cuda_module.cu源文件编译生成PTX代码
// HELLO_CUDA_MODULE: hello_cuda_module.cu源文件编译生成的PTX代码的字符串常量
use course02_hello_cuda_bindgen::HELLO_CUDA_MODULE;

// 普通函数（CPU主机端调用和执行）
fn hello_cuda_from_cpu() {
    println!("CPU: 你好, CUDA! (Rust版)");
}

fn main() -> Result<()> {
    // GPU: 你好, CUDA! (Rust版)
    {
        // 根据第0号CUDA设备，创建CUDA Device实例
        let dev = CudaDevice::new(0)?;
        // 从PTX中动态加载指定模块中的CUDA核函数集合
        dev.load_ptx(HELLO_CUDA_MODULE.into(), "hello_cuda_module", &["hello_cuda_from_gpu"])?;
        // 获取指定模块中的某个CUDA核函数实例
        let hello_cuda_from_gpu = dev.get_func("hello_cuda_module", "hello_cuda_from_gpu").unwrap();
        // CUDA核函数配置参数, 核函数总线程数为(x,y,z)=(16,1,1)
        let cfg = LaunchConfig {
            block_dim: (8,1,1),     // Block(线程块)大小（即1个Block中的线程数量）
            grid_dim: (2,1,1),      // Grid(网格)大小（即1个Grid中的线程块数量）
            shared_mem_bytes: 0,    // 共享内存大小
        };
        // CUDA核函数调用
        unsafe { hello_cuda_from_gpu.launch(cfg, (0,)) }?;
    }
    println!();
    // CPU: 你好, CUDA! (Rust版)
    {
        for _ in 0..16  {
            hello_cuda_from_cpu();
        }
    }
    Ok(())
}