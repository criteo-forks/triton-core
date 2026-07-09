// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dynamic_batch_scheduler.h"

#include "gtest/gtest.h"

namespace triton { namespace core {
namespace {

// The slot-wait window is clamped to the closest request deadline of the
// pending batch so that REJECT-policy timeouts are enforced promptly while
// the batcher is blocked on backpressure. These cover the pure deadline math.

constexpr uint64_t kDefaultUs = 500 * 1000;  // 500ms default window
constexpr uint64_t kFloorUs = 1000;          // 1ms anti-busy-poll floor
constexpr uint64_t kNowNs = 1'000'000'000'000ull;

// No timeout policy in effect: the default window must pass through
// untouched, preserving pre-change behavior.
TEST(ClampWaitToDeadlineTest, NoDeadlinePassesThrough)
{
  EXPECT_EQ(
      DynamicBatchScheduler::ClampWaitToDeadline(kDefaultUs, 0, kNowNs),
      kDefaultUs);
}

// A deadline further away than the default window must not extend the wait.
TEST(ClampWaitToDeadlineTest, FarDeadlineKeepsDefault)
{
  const uint64_t deadline_ns = kNowNs + 10ull * 1000 * 1000 * 1000;  // +10s
  EXPECT_EQ(
      DynamicBatchScheduler::ClampWaitToDeadline(
          kDefaultUs, deadline_ns, kNowNs),
      kDefaultUs);
}

// A deadline inside the window clamps the wait to the remaining time.
TEST(ClampWaitToDeadlineTest, NearDeadlineClampsWait)
{
  const uint64_t remaining_us = 20 * 1000;  // 20ms
  const uint64_t deadline_ns = kNowNs + remaining_us * 1000;
  EXPECT_EQ(
      DynamicBatchScheduler::ClampWaitToDeadline(
          kDefaultUs, deadline_ns, kNowNs),
      remaining_us);
}

// A deadline closer than the floor must not degenerate into a busy poll.
TEST(ClampWaitToDeadlineTest, FloorPreventsBusyPoll)
{
  const uint64_t deadline_ns = kNowNs + 10 * 1000;  // +10µs
  EXPECT_EQ(
      DynamicBatchScheduler::ClampWaitToDeadline(
          kDefaultUs, deadline_ns, kNowNs),
      kFloorUs);
}

// An already-expired deadline yields the minimal wait so the caller's next
// pass runs the reject scan promptly.
TEST(ClampWaitToDeadlineTest, ExpiredDeadlineYieldsMinimalWait)
{
  const uint64_t deadline_ns = kNowNs - 1;
  EXPECT_EQ(
      DynamicBatchScheduler::ClampWaitToDeadline(
          kDefaultUs, deadline_ns, kNowNs),
      kFloorUs);
  EXPECT_EQ(
      DynamicBatchScheduler::ClampWaitToDeadline(kDefaultUs, kNowNs, kNowNs),
      kFloorUs);
}

// The clamp never returns more than the caller's window nor less than the
// floor, across a sweep of deadlines.
TEST(ClampWaitToDeadlineTest, ResultAlwaysWithinBounds)
{
  for (uint64_t delta_us : {0ull, 1ull, 999ull, 1000ull, 250'000ull,
                            500'000ull, 750'000ull, 5'000'000ull}) {
    const uint64_t deadline_ns = kNowNs + delta_us * 1000;
    const uint64_t wait = DynamicBatchScheduler::ClampWaitToDeadline(
        kDefaultUs, deadline_ns, kNowNs);
    EXPECT_GE(wait, kFloorUs) << "delta_us=" << delta_us;
    EXPECT_LE(wait, kDefaultUs) << "delta_us=" << delta_us;
  }
}

}  // namespace
}}  // namespace triton::core
