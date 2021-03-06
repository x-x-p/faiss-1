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

BLOCK_SELECT_IMPL(int, true, 256, 4);
BLOCK_SELECT_IMPL(int, false, 256, 4);

} } // namespace
