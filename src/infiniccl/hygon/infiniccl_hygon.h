#ifndef INFINICCL_HYGON_H_
#define INFINICCL_HYGON_H_

#include "../infiniccl_impl.h"

// Windows does not support Hygon CCL
#if defined(ENABLE_HYGON_API) && defined(ENABLE_CCL) && !defined(_WIN32)
INFINICCL_DEVICE_API_IMPL(hygon)
#else
INFINICCL_DEVICE_API_NOOP(hygon)
#endif

#endif /* INFINICCL_HYGON_H_ */
