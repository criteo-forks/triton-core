// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include "constants.h"
#include "memory.h"
#include "metric_model_reporter.h"
#include "model_config.pb.h"
#include "model_config_utils.h"
#include "server_message.h"
#include "status.h"
#include "triton/common/sync_queue.h"

namespace triton { namespace core {

class TritonModel;
class InferenceRequest;

//
// Represents a model instance.
//
class TritonModelInstance {
 public:
  struct SecondaryDevice {
    SecondaryDevice(const std::string kind, const int64_t id)
        : kind_(kind), id_(id)
    {
    }
    const std::string kind_;
    const int64_t id_;
  };

  class Signature {
   public:
    Signature(
        const inference::ModelInstanceGroup& group_config, int32_t device_id)
        : group_config_(group_config), device_id_(device_id), can_match_(true)
    {
    }
    // Check if the lhs signature is equivalent to the rhs, if matching is
    // enabled. If matching is disabled, lhs != rhs under all scenarios.
    bool operator==(const Signature& rhs) const
    {
      return can_match_ && rhs.can_match_ && device_id_ == rhs.device_id_ &&
             EquivalentInInstanceConfig(group_config_, rhs.group_config_);
    }
    bool operator!=(const Signature& rhs) const { return !(*this == rhs); }
    // Enable/Disable matching. If disabled, on either lhs or rhs or both, then
    // lhs != rhs under all scenarios, including if they are equivalent.
    // This feature is intended to filter out signatures that have already been
    // matched, by disabling matching.
    void EnableMatching() { can_match_ = true; }
    void DisableMatching() { can_match_ = false; }

   private:
    const inference::ModelInstanceGroup group_config_;
    const int32_t device_id_;
    bool can_match_;  // cannot match another signature if false
  };

  static Status SetInstances(
      TritonModel* model,
      const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const triton::common::HostPolicyCmdlineConfigMap& host_policy_map,
      const inference::ModelConfig& model_config);
  ~TritonModelInstance();

  const std::string& Name() const { return name_; }
  Signature& GetSignature() { return signature_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }
  const triton::common::HostPolicyCmdlineConfig& HostPolicy() const
  {
    return host_policy_;
  }
  const TritonServerMessage& HostPolicyMessage() const
  {
    return host_policy_message_;
  }
  bool IsPassive() const { return passive_; }
  const std::vector<std::string>& Profiles() const { return profile_names_; }

  const std::vector<SecondaryDevice>& SecondaryDevices() const
  {
    return secondary_devices_;
  }

  Status Initialize();
  Status WarmUp();
  void Schedule(
      std::vector<std::unique_ptr<InferenceRequest>>&& requests,
      const std::function<void()>& OnCompletion);

  TritonModel* Model() const { return model_; }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  MetricModelReporter* MetricReporter() const { return reporter_.get(); }

 private:
  class TritonBackendThread {
   public:
    static Status CreateBackendThread(
        const std::string name, TritonModelInstance* model, const int nice,
        const int32_t device_id,
        std::unique_ptr<TritonBackendThread>* triton_backend_thread);
    void AddModelInstance(TritonModelInstance* model_instance);
    Status InitAndWarmUpModelInstance(TritonModelInstance* model_instance);
    void StopBackendThread();
    ~TritonBackendThread();

   private:
    TritonBackendThread(
        const std::string& name, TritonModel* model, const int nice,
        const int32_t device_id);
    void BackendThread();

    const std::string name_;
    const int nice_;
    const int32_t device_id_;

    TritonModel* model_;
    std::deque<TritonModelInstance*> model_instances_;

    std::thread backend_thread_;
    std::atomic<bool> backend_thread_exit_;
  };

  struct WarmupData {
    WarmupData(const std::string& sample_name, const size_t count)
        : sample_name_(sample_name), count_(std::max(count, size_t{1}))
    {
    }

    std::string sample_name_;
    size_t count_;
    // Using a batch of requests to satisfy batch size, this provides better
    // alignment on the batch expected by the model, especially for sequence
    // model.
    std::vector<std::unique_ptr<InferenceRequest>> requests_;

    // Placeholder for input data
    std::unique_ptr<AllocatedMemory> zero_data_;
    std::unique_ptr<AllocatedMemory> random_data_;
    std::vector<std::unique_ptr<std::string>> provided_data_;
  };

  DISALLOW_COPY_AND_ASSIGN(TritonModelInstance);
  TritonModelInstance(
      TritonModel* model, const std::string& name, const Signature& signature,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const triton::common::HostPolicyCmdlineConfig& host_policy,
      const TritonServerMessage& host_policy_message,
      const std::vector<SecondaryDevice>& secondary_devices);
  static Status CreateInstance(
      TritonModel* model, const std::string& name, const Signature& signature,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const std::string& host_policy_name,
      const triton::common::HostPolicyCmdlineConfig& host_policy,
      const inference::ModelRateLimiter& rate_limiter_config,
      const std::vector<SecondaryDevice>& secondary_devices,
      std::shared_ptr<TritonModelInstance>* triton_model_instance);
  Status SetBackendThread(
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const bool device_blocking);
  Status GenerateWarmupData();

  void Execute(std::vector<TRITONBACKEND_Request*>& triton_requests);

  std::shared_ptr<TritonBackendThread> triton_backend_thread_;

  std::vector<WarmupData> warmup_samples_;

  // The TritonModel object that owns this instance. The instance
  // holds this as a raw pointer because the lifetime of the model is
  // guaranteed to be longer than the lifetime of an instance owned by the
  // model.
  TritonModel* model_;

  std::string name_;
  Signature signature_;

  // For CPU device_id_ is always 0. For GPU device_id_ indicates the
  // GPU device to be used by the instance.
  TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;
  const triton::common::HostPolicyCmdlineConfig host_policy_;
  TritonServerMessage host_policy_message_;
  std::vector<std::string> profile_names_;
  bool passive_;

  std::vector<SecondaryDevice> secondary_devices_;

  // Reporter for metrics, or nullptr if no metrics should be reported
  std::shared_ptr<MetricModelReporter> reporter_;

  // Opaque state associated with this model instance.
  void* state_;
};

}}  // namespace triton::core
