// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <atomic>

#include "payload.h"

namespace triton { namespace core {

//
// InstanceQueue
//
// A queue implementation holding Payloads ready to be scheduled on
// model instance.
class InstanceQueue {
 public:
  explicit InstanceQueue(size_t max_batch_size, uint64_t max_queue_delay_ns);

  size_t Size();
  bool Empty();
  // Lock-free approximate size for advisory checks (e.g. the payload
  // prefetch cap); may lag concurrent enqueues/dequeues.
  size_t SizeApprox() const { return size_.load(std::memory_order_relaxed); }
  void Enqueue(const std::shared_ptr<Payload>& payload);
  void Dequeue(
      std::shared_ptr<Payload>* payload,
      std::vector<std::shared_ptr<Payload>>* merged_payloads);

  void IncrementConsumerCount();
  void DecrementConsumerCount();
  void WaitForConsumer();
  int WaitingConsumerCount();

 private:
  size_t max_batch_size_;
  uint64_t max_queue_delay_ns_;

  std::deque<std::shared_ptr<Payload>> payload_queue_;
  std::shared_ptr<Payload> staged_payload_;
  std::mutex mu_;
  // Mirrors payload_queue_.size(); maintained under 'mu_' by the callers'
  // locking discipline, readable without the lock via SizeApprox().
  std::atomic<size_t> size_{0};

  int waiting_consumer_count_;
  std::mutex waiting_consumer_mu_;
  std::condition_variable waiting_consumer_cv_;
};

}}  // namespace triton::core
