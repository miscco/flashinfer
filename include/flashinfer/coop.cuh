/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda/experimental/coop/algorithm>
#include <cuda/experimental/coop/group>
#include <cuda/hierarchy>

// Cooperative tile transfers of cuda::experimental::coop for kernels launched with <<<>>>.
namespace flashinfer::coop {

using namespace cuda::experimental::coop;

using LinearBlockHierarchy = decltype(cuda::make_hierarchy(cuda::block_dims(1u)));

// The warp of the calling thread in a one-dimensional block
__device__ __forceinline__ this_warp<LinearBlockHierarchy> thisWarp() {
  return this_warp{cuda::make_hierarchy(cuda::block_dims(blockDim.x))};
}

// The block of the calling thread, which must be one-dimensional
__device__ __forceinline__ this_block<LinearBlockHierarchy> thisBlock() {
  return this_block{cuda::make_hierarchy(cuda::block_dims(blockDim.x))};
}

}  // namespace flashinfer::coop
