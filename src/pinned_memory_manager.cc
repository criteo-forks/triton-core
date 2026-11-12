// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "pinned_memory_manager.h"

#include <sstream>

#include "numa_utils.h"
#include "triton/common/logging.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace core {

namespace {

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

Status
ParseIntOption(const std::string& msg, const std::string& arg, int* value)
{
  try {
    *value = std::stoi(arg);
  }
  catch (const std::invalid_argument& ia) {
    return Status(
        Status::Code::INVALID_ARG,
        msg + ": Can't parse '" + arg + "' to integer");
  }
  return Status::Success;
}

}  // namespace

std::unique_ptr<PinnedMemoryManager> PinnedMemoryManager::instance_;
uint64_t PinnedMemoryManager::pinned_memory_byte_size_ = 0;
std::mutex PinnedMemoryManager::allocated_buffer_mtx_;
std::vector<std::shared_ptr<PinnedMemoryManager::PinnedMemory>>
    PinnedMemoryManager::allocated_pinned_memory_buffers_;

PinnedMemoryManager::PinnedMemory::PinnedMemory(
    void* pinned_memory_buffer, uint64_t size)
    : pinned_memory_buffer_(pinned_memory_buffer),
      pinned_memory_buffer_size_((pinned_memory_buffer == nullptr) ? 0 : size)
{
  used_pinned_memory_byte_size_ = 0;
  if (pinned_memory_buffer_ != nullptr) {
    managed_pinned_memory_ = boost::interprocess::managed_external_buffer(
        boost::interprocess::create_only_t{}, pinned_memory_buffer_, size);
  }
}


PinnedMemoryManager::PinnedMemory::~PinnedMemory()
{
#ifdef TRITON_ENABLE_GPU
  if (pinned_memory_buffer_ != nullptr) {
    cudaFreeHost(pinned_memory_buffer_);
  }
#endif  // TRITON_ENABLE_GPU
}

void*
PinnedMemoryManager::PinnedMemory::Allocate(uint64_t size)
{
  std::lock_guard<std::mutex> lk(buffer_mtx_);
  void* ptr = managed_pinned_memory_.allocate(size, std::nothrow_t{});
  if (ptr != nullptr) {
    used_pinned_memory_byte_size_ +=
        managed_pinned_memory_.get_segment_manager()->size(ptr);
  }
  return ptr;
}

void
PinnedMemoryManager::PinnedMemory::Deallocate(void* ptr)
{
  std::lock_guard<std::mutex> lk(buffer_mtx_);
  used_pinned_memory_byte_size_ -=
      managed_pinned_memory_.get_segment_manager()->size(ptr);
  managed_pinned_memory_.deallocate(ptr);
}

bool
PinnedMemoryManager::PinnedMemory::Contains(const void* ptr) const
{
  const auto base = reinterpret_cast<uintptr_t>(pinned_memory_buffer_);
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  return (pinned_memory_buffer_ != nullptr) && (addr >= base) &&
         (addr < base + pinned_memory_buffer_size_);
}

uint64_t
PinnedMemoryManager::PinnedMemory::GetUsedPinnedMemorySizeInternal()
{
  std::lock_guard<std::mutex> lk(buffer_mtx_);
  return used_pinned_memory_byte_size_;
}

PinnedMemoryManager::~PinnedMemoryManager()
{
  // Clean up
  allocated_pinned_memory_buffers_.clear();
}

void
PinnedMemoryManager::AddPinnedMemoryBuffer(
    const std::shared_ptr<PinnedMemory>& pinned_memory_buffer,
    unsigned long node_mask)
{
  pinned_memory_buffers_[node_mask] = pinned_memory_buffer;
  {
    std::lock_guard<std::mutex> lk(allocated_buffer_mtx_);
    allocated_pinned_memory_buffers_.push_back(pinned_memory_buffer);
  }
}

Status
PinnedMemoryManager::AllocInternal(
    void** ptr, uint64_t size, TRITONSERVER_MemoryType* allocated_type,
    bool allow_nonpinned_fallback, PinnedMemory* pinned_memory_buffer)
{
  const bool has_pool =
      (pinned_memory_buffer->pinned_memory_buffer_ != nullptr);
  if (has_pool) {
    *ptr = pinned_memory_buffer->Allocate(size);
    if (*ptr != nullptr) {
      *allocated_type = TRITONSERVER_MEMORY_CPU_PINNED;
      LOG_VERBOSE(1) << "pinned memory allocation: "
                     << "size " << size << ", addr " << *ptr;
      return Status::Success;
    }
  }

  const char* pinned_error =
      has_pool
          ? "failed to allocate pinned system memory"
          : "failed to allocate pinned system memory: no pinned memory pool";
  if (!allow_nonpinned_fallback) {
    return Status(Status::Code::INTERNAL, pinned_error);
  }

  static bool warning_logged = false;
  if (!warning_logged) {
    LOG_WARNING << pinned_error << ", falling back to non-pinned system memory";
    warning_logged = true;
  }
  *ptr = malloc(size);
  *allocated_type = TRITONSERVER_MEMORY_CPU;
  if (*ptr == nullptr) {
    return Status(
        Status::Code::INTERNAL, "failed to allocate non-pinned system memory");
  }
  LOG_VERBOSE(1) << "non-pinned memory allocation: "
                 << "size " << size << ", addr " << *ptr;
  return Status::Success;
}

Status
PinnedMemoryManager::FreeInternal(void* ptr)
{
  for (const auto& pinned_memory_buffer : pinned_memory_buffers_) {
    if (pinned_memory_buffer.second->Contains(ptr)) {
      LOG_VERBOSE(1) << "pinned memory deallocation: "
                     << "addr " << ptr;
      pinned_memory_buffer.second->Deallocate(ptr);
      return Status::Success;
    }
  }

  LOG_VERBOSE(1) << "non-pinned memory deallocation: "
                 << "addr " << ptr;
  free(ptr);
  return Status::Success;
}

void
PinnedMemoryManager::Reset()
{
  instance_.reset();
}

Status
PinnedMemoryManager::Create(const Options& options)
{
  if (instance_ != nullptr) {
    LOG_WARNING << "New pinned memory pool of size "
                << options.pinned_memory_pool_byte_size_
                << " could not be created since one already exists"
                << " of size " << pinned_memory_byte_size_;
    return Status::Success;
  }

  instance_.reset(new PinnedMemoryManager());
  if (options.host_policy_map_.empty()) {
    void* buffer = nullptr;
    if (options.pinned_memory_pool_byte_size_ > 0) {
#ifdef TRITON_ENABLE_GPU
      auto err = cudaHostAlloc(
          &buffer, options.pinned_memory_pool_byte_size_,
          cudaHostAllocPortable);
      if (err != cudaSuccess) {
        buffer = nullptr;
        LOG_WARNING << "Unable to allocate pinned system memory, pinned memory "
                       "pool will not be available: "
                    << std::string(cudaGetErrorString(err));
      } else if (options.pinned_memory_pool_byte_size_ != 0) {
        LOG_INFO << "Pinned memory pool is created at '"
                 << PointerToString(buffer) << "' with size "
                 << options.pinned_memory_pool_byte_size_;
      }
#endif  // TRITON_ENABLE_GPU
    } else {
      LOG_INFO << "Pinned memory pool disabled";
    }
    try {
      instance_->AddPinnedMemoryBuffer(
          std::shared_ptr<PinnedMemory>(
              new PinnedMemory(buffer, options.pinned_memory_pool_byte_size_)),
          0);
    }
    catch (const std::exception& ex) {
      return Status(
          Status::Code::INTERNAL,
          "Failed to add Pinned Memory buffer: " + std::string(ex.what()));
    }
  } else {
    // Create only one buffer / manager should be created for one node,
    // and all associated devices should request memory from the shared manager
    std::map<int32_t, std::string> numa_map;
    for (const auto& host_policy : options.host_policy_map_) {
      const auto numa_it = host_policy.second.find("numa-node");
      if (numa_it != host_policy.second.end()) {
        int32_t numa_id;
        if (ParseIntOption("Parsing NUMA node", numa_it->second, &numa_id)
                .IsOk()) {
          numa_map.emplace(numa_id, host_policy.first);
        }
      }
    }
    for (const auto& node_policy : numa_map) {
      auto status =
          SetNumaMemoryPolicy(options.host_policy_map_.at(node_policy.second));
      if (!status.IsOk()) {
        LOG_WARNING << "Unable to allocate pinned system memory for NUMA node "
                    << node_policy.first << ": " << status.AsString();
        continue;
      }
      unsigned long node_mask;
      status = GetNumaMemoryPolicyNodeMask(&node_mask);
      if (!status.IsOk()) {
        LOG_WARNING << "Unable to get NUMA node set for current thread: "
                    << status.AsString();
        continue;
      }
      void* buffer = nullptr;
      if (options.pinned_memory_pool_byte_size_ > 0) {
#ifdef TRITON_ENABLE_GPU
        auto err = cudaHostAlloc(
            &buffer, options.pinned_memory_pool_byte_size_,
            cudaHostAllocPortable);
        if (err != cudaSuccess) {
          buffer = nullptr;
          LOG_WARNING
              << "Unable to allocate pinned system memory, pinned memory "
                 "pool will not be available: "
              << std::string(cudaGetErrorString(err));
        } else if (options.pinned_memory_pool_byte_size_ != 0) {
          LOG_INFO << "Pinned memory pool is created at '"
                   << PointerToString(buffer) << "' with size "
                   << options.pinned_memory_pool_byte_size_;
        } else {
          LOG_INFO << "Pinned memory pool disabled";
        }
#endif  // TRITON_ENABLE_GPU
      } else {
        LOG_INFO << "Pinned memory pool disabled";
      }
      ResetNumaMemoryPolicy();
      try {
        instance_->AddPinnedMemoryBuffer(
            std::shared_ptr<PinnedMemory>(new PinnedMemory(
                buffer, options.pinned_memory_pool_byte_size_)),
            node_mask);
      }
      catch (const std::exception& ex) {
        return Status(
            Status::Code::INTERNAL,
            "Failed to add Pinned Memory buffer with host policy: " +
                std::string(ex.what()));
      }
      // If no pinned memory is allocated, add an empty entry where all
      // allocation will be on normal system memory
      if (instance_->pinned_memory_buffers_.empty()) {
        try {
          instance_->AddPinnedMemoryBuffer(
              std::shared_ptr<PinnedMemory>(new PinnedMemory(
                  nullptr, options.pinned_memory_pool_byte_size_)),
              0);
        }
        catch (const std::exception& ex) {
          return Status(
              Status::Code::INTERNAL,
              "Failed to add empty Pinned Memory entry: " +
                  std::string(ex.what()));
        }
      }
    }
  }
  pinned_memory_byte_size_ = options.pinned_memory_pool_byte_size_;
  return Status::Success;
}

Status
PinnedMemoryManager::Alloc(
    void** ptr, uint64_t size, TRITONSERVER_MemoryType* allocated_type,
    bool allow_nonpinned_fallback)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "PinnedMemoryManager has not been created");
  }

  auto pinned_memory_buffer =
      instance_->pinned_memory_buffers_.begin()->second.get();
  if (instance_->pinned_memory_buffers_.size() > 1) {
    unsigned long node_mask;
    if (GetNumaMemoryPolicyNodeMask(&node_mask).IsOk()) {
      auto it = instance_->pinned_memory_buffers_.find(node_mask);
      if (it != instance_->pinned_memory_buffers_.end()) {
        pinned_memory_buffer = it->second.get();
      }
    }
  }

  return instance_->AllocInternal(
      ptr, size, allocated_type, allow_nonpinned_fallback,
      pinned_memory_buffer);
}

Status
PinnedMemoryManager::Free(void* ptr)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "PinnedMemoryManager has not been created");
  }

  return instance_->FreeInternal(ptr);
}

uint64_t
PinnedMemoryManager::GetTotalPinnedMemoryByteSize()
{
  return pinned_memory_byte_size_;
}

uint64_t
PinnedMemoryManager::GetUsedPinnedMemoryByteSize()
{
  std::lock_guard<std::mutex> lk(allocated_buffer_mtx_);
  uint64_t used_pinned_memory_size = 0;
  if (!allocated_pinned_memory_buffers_.empty()) {
    for (const auto& it : allocated_pinned_memory_buffers_) {
      used_pinned_memory_size += it->GetUsedPinnedMemorySizeInternal();
    }
  }

  return used_pinned_memory_size;
}

}}  // namespace triton::core
