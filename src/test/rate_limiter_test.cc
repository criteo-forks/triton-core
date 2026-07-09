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

#include "rate_limiter.h"

#include <memory>

#include "gtest/gtest.h"

namespace triton { namespace core {
namespace {

std::unique_ptr<RateLimiter>
MakeRateLimiter()
{
  std::unique_ptr<RateLimiter> rate_limiter;
  auto status = RateLimiter::Create(
      true /* ignore_resources_and_priority */, RateLimiter::ResourceMap(),
      &rate_limiter);
  EXPECT_TRUE(status.IsOk());
  return rate_limiter;
}

// The payload queue handle used by hot-path callers to skip the global map
// lookup: an unregistered model must yield nullptr, not a crash or a
// fabricated entry.
TEST(RateLimiterTest, LookupPayloadQueueUnknownModelIsNull)
{
  auto rate_limiter = MakeRateLimiter();
  EXPECT_EQ(
      rate_limiter->LookupPayloadQueue(nullptr /* unregistered model */),
      nullptr);
}

// Callers passing no cached handle rely on the internal fallback lookup;
// enqueuing for an unregistered model must surface an error, exactly as the
// pre-caching code did.
TEST(RateLimiterTest, EnqueuePayloadUnknownModelFails)
{
  auto rate_limiter = MakeRateLimiter();
  auto payload =
      rate_limiter->GetPayload(Payload::Operation::INFER_RUN, nullptr);
  auto status =
      rate_limiter->EnqueuePayload(nullptr /* unregistered model */, payload);
  EXPECT_FALSE(status.IsOk());
}

// The slot probe must degrade to "no slot" (never block, never crash) when
// the model has no payload queue yet, for both prefetching modes.
TEST(RateLimiterTest, PayloadSlotAvailableUnknownModelIsFalse)
{
  auto rate_limiter = MakeRateLimiter();
  EXPECT_FALSE(rate_limiter->PayloadSlotAvailable(
      nullptr /* unregistered model */, nullptr /* instance */,
      true /* support_prefetching */, true /* force_non_blocking */));
  EXPECT_FALSE(rate_limiter->PayloadSlotAvailable(
      nullptr /* unregistered model */, nullptr /* instance */,
      false /* support_prefetching */, true /* force_non_blocking */));
}

}  // namespace
}}  // namespace triton::core
