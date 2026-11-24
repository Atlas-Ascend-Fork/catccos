# Match communication lowering and both device templates to the SHMEM build.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_shmem_build "${TLA_DIST_DSL_SHMEM_ROOT}/build")
set(_shmem_config_reader "${TLA_DIST_DSL_ROOT}/tools/read_shmem_rdma_config.py")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${_shmem_config_reader}" "${TLA_DIST_DSL_SHMEM_ROOT}"
  OUTPUT_VARIABLE _shmem_rdma_config
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${_shmem_build}/CMakeCache.txt" "${_shmem_build}/compile_commands.json"
             "${_shmem_config_reader}")
string(JSON _shmem_backend GET "${_shmem_rdma_config}" rdma_backend)
set(TLA_DIST_RDMA_BACKEND "${_shmem_backend}" CACHE INTERNAL "RDMA backend from SHMEM" FORCE)
set(TLA_DIST_ENABLE_RDMA 0)
set(TLA_DIST_RDMA_COMPILE_OPTIONS)
if(TLA_DIST_RDMA_BACKEND STREQUAL "XSCALE")
  set(TLA_DIST_ENABLE_RDMA 1)
  list(APPEND TLA_DIST_RDMA_COMPILE_OPTIONS -DACLSHMEMI_RDMA_K_BACKEND_XSCALE=1)
elseif(TLA_DIST_RDMA_BACKEND STREQUAL "HNS_1825")
  string(JSON _shmem_pi_on_chip GET "${_shmem_rdma_config}" pi_on_chip)
  message(STATUS "SHMEM HNS_1825 PI_ON_CHIP: ${_shmem_pi_on_chip}")
  set(TLA_DIST_ENABLE_RDMA 1)
  list(APPEND TLA_DIST_RDMA_COMPILE_OPTIONS
       -DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1
       "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=${_shmem_pi_on_chip}")
endif()
message(STATUS "CATCCOS DSL RDMA backend: ${TLA_DIST_RDMA_BACKEND} (from ${_shmem_build})")
