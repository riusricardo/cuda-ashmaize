use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("CUDA initialization failed")]
    InitFailed,
    
    #[error("No CUDA-capable GPU found")]
    NoDevice,
    
    #[error("ROM not uploaded to GPU")]
    NoRom,
    
    #[error("CUDA memory allocation failed")]
    AllocationFailed,
    
    #[error("CUDA kernel launch failed")]
    KernelLaunchFailed,
    
    #[error("CUDA memory transfer failed")]
    MemoryTransferFailed,
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("CUDA error: {0}")]
    CudaError(String),
}

pub type Result<T> = std::result::Result<T, GpuError>;
