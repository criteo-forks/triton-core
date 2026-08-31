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
//
#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace triton { namespace core {

/// Readiness status for models.
enum class ModelReadyState {
  // The model is in an unknown state. The model is not available for
  // inferencing.
  UNKNOWN,

  // The model is ready and available for inferencing.
  READY,

  // The model is unavailable, indicating that the model failed to
  // load or has been implicitly or explicitly unloaded. The model is
  // not available for inferencing.
  UNAVAILABLE,

  // The model is being loaded by the inference server. The model is
  // not available for inferencing.
  LOADING,

  // The model is being unloaded by the inference server. The model is
  // not available for inferencing.
  UNLOADING
};

/// Get the string representation for a ModelReadyState
const std::string& ModelReadyStateString(ModelReadyState state);

// Stable, machine-checkable prefixes for the repository-index 'reason' of a
// version whose unload cannot complete. Part of the API contract documented
// at TRITONSERVER_ServerModelReportLeakedReference in tritonserver.h - do
// not reword without updating that documentation and downstream consumers.
inline constexpr char kStuckUnloadReasonPrefix[] = "stuck: ";
inline constexpr char kLeakedReferenceWarningPrefix[] = "warning: ";

// Decide how a model version is reported in the repository index. Pure
// function, exposed for unit testing; the caller supplies:
//   state / state_reason  - the raw lifecycle values,
//   leaked_refs           - reference leaks reported for the version,
//   unloading_for_ns      - time spent in UNLOADING, or -1 if unknown,
//   stuck_threshold_ns    - UNLOADING longer than this is stuck (<=0
//                           disables the timeout),
//   live_refs             - model references currently held.
// The state is never changed - only the reason is annotated.
inline std::pair<ModelReadyState, std::string>
ReportedModelState(
    const ModelReadyState state, const std::string& state_reason,
    const uint64_t leaked_refs, const int64_t unloading_for_ns,
    const int64_t stuck_threshold_ns, const long live_refs)
{
  if (leaked_refs > 0) {
    const std::string leaked = std::to_string(leaked_refs);
    if (state == ModelReadyState::UNLOADING) {
      return std::make_pair(
          state, kStuckUnloadReasonPrefix + leaked +
                     " leaked reference(s) (dropped reply hand-offs); unload "
                     "cannot complete; resident until process restart");
    }
    if (state == ModelReadyState::READY) {
      return std::make_pair(
          state, kLeakedReferenceWarningPrefix + leaked +
                     " leaked reference(s) (dropped reply hand-offs); unload "
                     "of this version will stick until process restart");
    }
  }
  if ((state == ModelReadyState::UNLOADING) && (stuck_threshold_ns > 0) &&
      (unloading_for_ns >= stuck_threshold_ns)) {
    const int64_t stuck_s = unloading_for_ns / 1000000000LL;
    const std::string held =
        (live_refs > 0)
            ? (std::to_string(live_refs) + " reference(s) still held")
            : "references drained but finalization has not completed";
    return std::make_pair(
        state, kStuckUnloadReasonPrefix + std::string("unload requested ") +
                   std::to_string(stuck_s) + "s ago; " + held +
                   "; resident until process restart");
  }
  return std::make_pair(state, state_reason);
}

}}  // namespace triton::core
