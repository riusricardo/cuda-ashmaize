use std::os::raw::c_void;

// C FFI declarations for CUDA functions
extern "C" {
    pub fn gpu_init() -> i32;
    pub fn gpu_cleanup() -> i32;
    
    pub fn gpu_upload_rom(
        rom_data: *const u8,
        rom_size: usize,
        rom_digest: *const u8,
    ) -> *mut c_void;
    
    pub fn gpu_mine_batch(
        rom_handle: *mut c_void,
        salts: *const u8,
        hashes: *mut u8,
        flags: *mut u8,
        batch_size: u32,
        salt_len: u32,
        nb_loops: u32,
        nb_instrs: u32,
    ) -> i32;
    
    pub fn gpu_free_rom(rom_handle: *mut c_void);
}
