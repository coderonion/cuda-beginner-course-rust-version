use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use anyhow::Result;

const PTX_SRC: &str = r#"
        extern "C" __global__ void hello_cuda_from_gpu(size_t num) {
            printf("GPU: 你好, CUDA! (Rust版)\n");
        }
    "#;

fn main() -> Result<()> {
    // GPU: 你好, CUDA! (Rust版)
    {
        let dev = cudarc::driver::CudaDevice::new(0)?;
        let ptx = cudarc::nvrtc::compile_ptx(PTX_SRC)?;
        dev.load_ptx(ptx, "hello_cuda_from_gpu", &["hello_cuda_from_gpu"])?;
        let hello_cuda_from_gpu = dev.get_func("hello_cuda_from_gpu", "hello_cuda_from_gpu").unwrap();
        let cfg = LaunchConfig {
            block_dim: (1,1,9),
            grid_dim: (1,1,2),
            shared_mem_bytes: 0,
        };
        unsafe { hello_cuda_from_gpu.launch(cfg, (0usize,)) }?;
    }
    println!();
    // CPU: 你好, CUDA! (Rust版)
    {
        for _ in 0..16  {
            println!("CPU: 你好, CUDA! (Rust版)");
        }
    }
    Ok(())
}