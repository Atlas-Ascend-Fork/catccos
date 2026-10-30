set(TLA_DIST_AIV_BITCODE
    "${CMAKE_CURRENT_BINARY_DIR}/shmem.aiv.c310.bc")

tla_dist_compile_vector_bitcode(
  "${CMAKE_CURRENT_SOURCE_DIR}/shmem.cpp" aiv "${TLA_DIST_AIV_BITCODE}")
set(TLA_DIST_MIX_AIV_BITCODE
    "${CMAKE_CURRENT_BINARY_DIR}/shmem.mix_aiv.c310.bc")
tla_dist_compile_vector_bitcode(
  "${CMAKE_CURRENT_SOURCE_DIR}/shmem.cpp" mix_aiv "${TLA_DIST_MIX_AIV_BITCODE}")

# A custom command's OUTPUT must be consumed by a target in the same CMake
# directory. Keep the raw bitcode rules reachable from the generated Ninja
# graph before the parent directory packages their files.
add_custom_target(tla_dist_shmem_vector_bitcode
  DEPENDS "${TLA_DIST_AIV_BITCODE}" "${TLA_DIST_MIX_AIV_BITCODE}")

set(TLA_DIST_AIV_BITCODE "${TLA_DIST_AIV_BITCODE}" PARENT_SCOPE)
set(TLA_DIST_MIX_AIV_BITCODE "${TLA_DIST_MIX_AIV_BITCODE}" PARENT_SCOPE)
