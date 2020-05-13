/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>

namespace faiss { namespace gpu {

#if GPU_MAX_SELECTION_K >= 2048
BLOCK_SELECT_IMPL(int, false, 2048, 8);
BLOCK_SELECT_IMPL(int, true, 2048, 8);
#endif

} } // namespace
