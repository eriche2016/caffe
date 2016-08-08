#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

// Caffe的底层数据的切换（cpu模式和gpu模式），需要用到内存同步模块
// data in disk -> cpu <-> GPU -> cpu -> disk to store data

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.

// 内联函数：分配内存
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {

#ifndef CPU_ONLY    //允许使用GPU
  if (Caffe::mode() == Caffe::GPU) { // GPU中调用 cudaMallocHost分配显存
    CUDA_CHECK(cudaMallocHost(ptr, size));  // 大小：size in bytes
    *use_cuda = true;            // use_cuda flag 设为true
    return;  // return 
  }
#endif  
  *ptr = malloc(size); // 在CPU内存中分配内存， 大小: size in bytes  
  *use_cuda = false;   // use_cuda flag设为false
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

// 内联函数：释放内存， ptr只想内存位置， use_cuda是一个bool值
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) { // 如果使用了cuda, 即use_cuda使true, 则释放显存
    CUDA_CHECK(cudaFreeHost(ptr)); // 释放显存
    return; // 返回
  }
#endif
  free(ptr); // 释放内存
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
 
 // SyncedMemory类
class SyncedMemory {
 public:
  // 默认建构子 
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  // 显式建构子
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  // 解构子
  ~SyncedMemory();
  
  const void* cpu_data(); // 返回的指针是const pointer， 故而不可以修改数据
  void set_cpu_data(void* data);
  const void* gpu_data(); // 返回的指针是const pointer， 故而不可以修改数据
  void set_gpu_data(void* data);
  void* mutable_cpu_data(); // 返回的指针没有const修饰，可以修改指针对应的数据
  void* mutable_gpu_data(); // 返回的指针没有const修饰，可以修改指针对应的数据
  
  /* url: http://www.jianshu.com/p/b105578b214b
  以to_cpu()方法为例: 检查head_所处状态, 若UNINITIALIZED, 则分配内存空间(置0); 
  若HEAD_AT_GPU, 则需要从GPU内存同步数据到CPU; HEAD_AT_CPU, 则说明目前最新的数
  据是在CPU的, 无须进行任何操作(虽然并不知道GPU的数据是否和CPU一致, 因为当前我
  们并不关心GPU数据); 若SYNCED, 则CPU/GPU数据一致, 无须进行任何操作.
  */
  // 枚举类型
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  // 返回数据状态：head_
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();
  void to_gpu();
  //数据在cpu或gpu，指向数据的指针
  void* cpu_ptr_;
  void* gpu_ptr_;
  
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
