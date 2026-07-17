// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <deque>
#include <memory>
#include <mutex>

namespace triton { namespace core {

//
// StagingQueue
//
// A small FIFO staging area that lets many producer threads hand off
// items to a single consumer thread without contending for the
// consumer's own (potentially long-held) lock. Producers only ever
// contend with each other and with the consumer's TakeAll() call, and
// each of those critical sections is a single deque operation.
template <typename ItemT>
class StagingQueue {
 public:
  // Append an item. Small critical section: one deque emplace.
  void Push(std::unique_ptr<ItemT>&& item)
  {
    std::lock_guard<std::mutex> lock(mu_);
    items_.emplace_back(std::move(item));
    size_.fetch_add(1, std::memory_order_relaxed);
  }

  // Remove and return all staged items in FIFO order.
  std::deque<std::unique_ptr<ItemT>> TakeAll()
  {
    std::deque<std::unique_ptr<ItemT>> taken;
    std::lock_guard<std::mutex> lock(mu_);
    taken.swap(items_);
    size_.fetch_sub(taken.size(), std::memory_order_relaxed);
    return taken;
  }

  // Lock-free approximate size. Advisory only: a producer's Push() is
  // only reflected here once it has acquired 'mu_' and completed the
  // fetch_add() below, and a concurrent TakeAll() may have already
  // removed items that are still counted for a brief window before its
  // fetch_sub() runs. Callers relying on Size()/Empty() for wake or
  // batching decisions must tolerate a stale answer; exactness is only
  // guaranteed while holding 'mu_' (e.g. immediately after TakeAll()).
  size_t Size() const { return size_.load(std::memory_order_relaxed); }
  bool Empty() const { return Size() == 0; }

  // Exact emptiness check that takes the internal mutex. Unlike Empty(),
  // this synchronizes with a concurrent Push(): once a producer's Push()
  // has returned, a subsequent EmptyExact() cannot report empty. Used by
  // the batcher's parking decision to close the lost-wakeup window.
  bool EmptyExact() const
  {
    std::lock_guard<std::mutex> lock(mu_);
    return items_.empty();
  }

 private:
  mutable std::mutex mu_;
  std::deque<std::unique_ptr<ItemT>> items_;
  std::atomic<size_t> size_{0};
};

}}  // namespace triton::core
