if(BISHENGIR_BUILD_TEMPLATE)
  if(NOT BISHENG_COMPILER_PATH)
    set(BISHENG_COMPILER_PATH "${ASCEND_HOME_PATH}/bin")
  endif()
  set(TLA_DIST_BISHENG_TOOL "${BISHENG_COMPILER_PATH}/ccec")
  set(TLA_DIST_LLVM_LINK_TOOL "${BISHENG_COMPILER_PATH}/llvm-link")
  if(NOT EXISTS "${TLA_DIST_BISHENG_TOOL}" OR
     NOT EXISTS "${TLA_DIST_LLVM_LINK_TOOL}")
    message(FATAL_ERROR
      "Cannot find ccec or llvm-link under ${BISHENG_COMPILER_PATH}")
  endif()

  function(tla_dist_compile_vector_bitcode source core_type output)
    set(bisheng_args
      -O2
      -x cce
      --cce-auto-sync=off
      --cce-aicore-only
      --cce-generic-addrspace=off
      "${source}"
      -emit-llvm
      -c
      -mllvm -disable-llvm-optzns
      -DCATLASS_ARCH=3510
      -DTILING_KEY_VAR
      -DL2_CACHE_HINT
      -Wno-ignored-attributes
      "-I${ASCEND_HOME_PATH}/include"
      "-I${ASCEND_HOME_PATH}/compiler/tikcpp"
      "-I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
      "-I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl"
      "-I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
      "-I${TLA_DIST_CATLASS_INCLUDE_DIR}"
      "-I${CATCCOS_ROOT}/include"
      "-I${CATLASS_TLA_DSL_ROOT}/csrc/mlir/bc"
      "-I${TLA_DIST_DSL_SHMEM_ROOT}/include"
      "-I${TLA_DIST_DSL_SHMEM_ROOT}/src"
      "-I${TLA_DIST_DSL_SHMEM_ROOT}/src/device"
      --cce-enable-print
      --cce-aicore-arch=dav-c310-vec
    )
    if(core_type STREQUAL "mix_aiv")
      # These are plain C ABI helper definitions, not a mixed kernel entry.
      # Select the MIX header implementations (including SyncAll<true>), but
      # leave MIX splitting/codegen to the final HIVMC invocation. Enabling
      # ccec MIX codegen here can make the unsuffixed helper exports unavailable
      # to the calls emitted by TlaDistLowerCommPass.
      list(APPEND bisheng_args -D__CCE_AICORE_ENABLE_MIX__=1
           -DTLA_DIST_REQUIRE_MIX_SYNC)
    elseif(NOT core_type STREQUAL "aiv")
      message(FATAL_ERROR "Unsupported TLA_DIST vector core type: ${core_type}")
    endif()
    list(APPEND bisheng_args -std=c++17 -o "${output}")

    add_custom_command(
      OUTPUT "${output}"
      COMMAND "${TLA_DIST_BISHENG_TOOL}" ${bisheng_args}
      COMMAND_EXPAND_LISTS
      DEPENDS
        "${source}"
        "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/bc/common.h"
        "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/bc/dma_common.h"
        "${CATCCOS_ROOT}/include/catccos/comm/tile/tile_remote_copy.hpp"
        "${CATCCOS_ROOT}/include/catccos/detail/remote_copy_type.hpp"
        "${TLA_DIST_DSL_SHMEM_ROOT}/include/shmem.h"
        "${TLA_DIST_DSL_SHMEM_ROOT}/src/device/gm2gm/shmemi_device_cc.h"
      VERBATIM
    )
  endfunction()

  add_subdirectory(Vector)

  set(_tla_dist_bc_package_dir "${TLA_DIST_PYTHON_OUTPUT_DIR}/bc")
  set(_tla_dist_base_dir "${CMAKE_CURRENT_BINARY_DIR}/base")
  set(_tla_dist_base_aic "${_tla_dist_base_dir}/meta_op.aic.c310.bc")
  set(_tla_dist_base_aiv "${_tla_dist_base_dir}/meta_op.aiv.c310.bc")
  file(GLOB_RECURSE _tla_dist_base_sources CONFIGURE_DEPENDS
    "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/bc/*.cpp"
    "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/bc/*.h"
    "${TLA_DIST_CATLASS_INCLUDE_DIR}/*.hpp")
  add_custom_command(
    OUTPUT "${_tla_dist_base_aic}" "${_tla_dist_base_aiv}"
    COMMAND "${Python3_EXECUTABLE}"
            "${TLA_DIST_DSL_ROOT}/tools/build_base_bitcode.py"
            --catlass-dsl-root "${CATLASS_TLA_DSL_ROOT}"
            --ascend-home "${ASCEND_HOME_PATH}"
            --compiler-bin "${BISHENG_COMPILER_PATH}"
            --output-dir "${_tla_dist_base_dir}"
    DEPENDS
      "${TLA_DIST_DSL_ROOT}/tools/build_base_bitcode.py"
      "${CATLASS_TLA_DSL_ROOT}/catlass/bc_compile.py"
      "${TLA_DIST_BISHENG_TOOL}"
      ${_tla_dist_base_sources}
    VERBATIM)
  add_custom_target(tla_dist_base_bitcode
    DEPENDS "${_tla_dist_base_aic}" "${_tla_dist_base_aiv}")
  set(_tla_dist_linked_aiv
      "${CMAKE_CURRENT_BINARY_DIR}/meta_op.tla_dist.aiv.c310.bc")
  set(_tla_dist_linked_mix_aiv
      "${CMAKE_CURRENT_BINARY_DIR}/meta_op.tla_dist.mix_aiv.c310.bc")
  # A mixed HIVMC invocation has exactly one template slot per core type. Link
  # the communication device functions into the regular AIV template before
  # packaging; passing SHMEM as a third comma-separated bitcode file leaves it
  # outside the mixed AIV slot and produces undefined symbols.
  add_custom_command(
    OUTPUT "${_tla_dist_linked_aiv}"
    COMMAND "${TLA_DIST_LLVM_LINK_TOOL}"
            "${_tla_dist_base_aiv}"
            "${TLA_DIST_AIV_BITCODE}"
            -o "${_tla_dist_linked_aiv}"
    DEPENDS
      tla_dist_base_bitcode
      tla_dist_shmem_vector_bitcode
      "${_tla_dist_base_aiv}"
      "${TLA_DIST_AIV_BITCODE}"
    VERBATIM
  )
  add_custom_target(tla_dist_linked_aiv_bitcode DEPENDS "${_tla_dist_linked_aiv}")
  add_custom_command(
    OUTPUT "${_tla_dist_linked_mix_aiv}"
    COMMAND "${TLA_DIST_LLVM_LINK_TOOL}"
            "${_tla_dist_base_aiv}"
            "${TLA_DIST_MIX_AIV_BITCODE}"
            -o "${_tla_dist_linked_mix_aiv}"
    DEPENDS
      tla_dist_base_bitcode
      tla_dist_shmem_vector_bitcode
      "${_tla_dist_base_aiv}"
      "${TLA_DIST_MIX_AIV_BITCODE}"
    VERBATIM
  )
  add_custom_target(tla_dist_linked_mix_aiv_bitcode
    DEPENDS "${_tla_dist_linked_mix_aiv}")
  set(_tla_dist_packaged_base_aic
      "${_tla_dist_bc_package_dir}/meta_op.aic.c310.bc")
  set(_tla_dist_packaged_base_aiv
      "${_tla_dist_bc_package_dir}/meta_op.aiv.c310.bc")
  set(_tla_dist_packaged_aiv "${_tla_dist_bc_package_dir}/shmem.aiv.c310.bc")
  set(_tla_dist_packaged_base_mix_aiv
      "${_tla_dist_bc_package_dir}/meta_op.mix_aiv.c310.bc")
  set(_tla_dist_packaged_mix_aiv
      "${_tla_dist_bc_package_dir}/shmem.mix_aiv.c310.bc")
  add_custom_command(
    OUTPUT
      "${_tla_dist_packaged_base_aic}"
      "${_tla_dist_packaged_base_aiv}"
      "${_tla_dist_packaged_aiv}"
      "${_tla_dist_packaged_base_mix_aiv}"
      "${_tla_dist_packaged_mix_aiv}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_tla_dist_bc_package_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_tla_dist_base_aic}" "${_tla_dist_packaged_base_aic}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_tla_dist_linked_aiv}" "${_tla_dist_packaged_base_aiv}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TLA_DIST_AIV_BITCODE}" "${_tla_dist_packaged_aiv}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_tla_dist_linked_mix_aiv}" "${_tla_dist_packaged_base_mix_aiv}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TLA_DIST_MIX_AIV_BITCODE}" "${_tla_dist_packaged_mix_aiv}"
    DEPENDS
      tla_dist_base_bitcode
      tla_dist_shmem_vector_bitcode
      tla_dist_linked_aiv_bitcode
      tla_dist_linked_mix_aiv_bitcode
      "${_tla_dist_base_aic}"
      "${_tla_dist_linked_aiv}"
      "${TLA_DIST_AIV_BITCODE}"
      "${_tla_dist_linked_mix_aiv}"
      "${TLA_DIST_MIX_AIV_BITCODE}"
  )
  add_custom_target(tla_dist_shmem_bitcode
    DEPENDS
      "${_tla_dist_packaged_base_aic}"
      "${_tla_dist_packaged_base_aiv}"
      "${_tla_dist_packaged_aiv}"
      "${_tla_dist_packaged_base_mix_aiv}"
      "${_tla_dist_packaged_mix_aiv}")
endif()
