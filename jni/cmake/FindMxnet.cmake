find_package(PkgConfig)
pkg_check_modules(PC_MXNET QUIET mxnet)

find_path(MXNET_INCLUDE_DIR mxnet-cpp/MxNetCpp.h
        PATH_SUFFIXES include)

find_library(MXNET_LIBRARY NAMES mxnet libmxnet
        PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(Mxnet DEFAULT_MSG
        MXNET_LIBRARY MXNET_INCLUDE_DIR)

set(MXNET_LIBRARIES ${MXNET_LIBRARY})
set(MXNET_INCLUDE_DIRS ${MXNET_INCLUDE_DIR})