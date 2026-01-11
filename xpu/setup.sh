
sudo gpasswd -a ${USER} render
newgrp render

# @yu recommends 
# sudo usermod -aG render,video $USER

- dump utilization
sudo xpu-smi dump -d 0,1,2,3 -m 18

https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
Verify installation and environment
source /opt/intel/oneapi/setvars.sh
sycl-ls


xpu-smi ps -h
xpu-smi ps


# 

# Install XPU driver:​
sudo apt-get update​
sudo apt-get install -y software-properties-common​
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics​
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc​
sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo​
Check XPU driver:​
clinfo | grep "Device Name"
Add user permission​
sudo usermod -aG render,video $USER​


# mamba

# curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# bash Miniforge3-$(uname)-$(uname -m).sh

# QBITS

# export ICPX_COMPILER_HOME=/opt/intel/oneapi/compiler/2025.2/bin/icpx
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.2/bin/icpx
# cmake --build build -j 8
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ICPX_COMPILER_HOME -DBTLA_UT_DEBUG=ON -DBTLA_UT_BENCHMARK=ON -DBTLA_SYCL=ON
# cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ICPX_COMPILER_HOME -DBTLA_UT_DEBUG=ON 
# cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ICPX_COMPILER_HOME -DBTLA_UT_DEBUG=ON 
# TLA
export ICPX_COMPILER_HOME=/opt/intel/oneapi/compiler/2025.2/bin/icpx
cmake -B build -DCMAKE_CXX_COMPILER=$ICPX_COMPILER_HOME -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS="-ftemplate-backtrace-limit=0 -fdiagnostics-color=always"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
export IGC_VISAOptions="-perfmodel"

export IGC_VectorAliasBBThreshold=10000

export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"

cmake --build build -j 8

# for torch?
# ONEAPI_DEVICE_SELECTOR=level_zero:0
# worked for sycl-kernel
# export ZE_AFFINITY_MASK=4,5,6,7
# Bfloat16 GEMM​
# ./examples/00_bmg_gemm/00_bmg_gemm_padded  --m=4096 --n=4096 --k=4096 --l=16 --iterations=100 ​
# Int4 WOQ GEMM:​
# ./examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_f16  --m=4096 --n=4096 --k=4096 --l=16 --iterations=10​
# Benchmark INT4/INT8​
# ./benchmarks/gemm/cutlass_benchmarks_gemm_sycl --config_file=../benchmarks/device/bmg/input_files/input_sglang_gemm_mixed_dtype.in​

# profiling
unitrace -d ./auto_round_extension/qbits/build/bestla/bestla_benchmark


=== Device Timing Summary ===

                Total Execution Time (ns):          22772555167
    Total Device Time for L0 backend (ns):           8858150990

== L0 Backend ==

                                                                                                                                                                                                                                                              Kernel,        Calls,            Time (ns),     Time (%),         Average (ns),             Min (ns),             Max (ns)
"gemv_v2(float const*, unsigned char const*, float const*, float*, int, int, int, int, int, sycl::_V1::queue*, sycl::_V1::range<1>, sycl::_V1::range<1>)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",        37374,           8816659656,    99.531601,               235903,               233750,              1016666
                                                                                                                                                                                                                                "zeCommandListAppendMemoryCopy(M2D)",           30,             41472851,     0.468189,              1382428,                 3541,              7503750
                                                                                                                                                                                                                                "zeCommandListAppendMemoryCopy(D2M)",           10,                18483,     0.000209,                 1848,                 1718,                 2031


=== Kernel Properties ===

                                                                                                                                                                                                                                                              Kernel, Compiled, SIMD, Number of Arguments, SLM Per Work Group, Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
"gemv_v2(float const*, unsigned char const*, float const*, float*, int, int, int, int, int, sycl::_V1::queue*, sycl::_V1::range<1>, sycl::_V1::range<1>)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",      JIT,   16,                   7,                  0,                         0,                       0,                           128


BesTLA Benchmark done

=== Device Timing Summary ===

                Total Execution Time (ns):          34856202187
    Total Device Time for L0 backend (ns):           4553583158

== L0 Backend ==

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Kernel,        Calls,            Time (ns),     Time (%),         Average (ns),             Min (ns),             Max (ns)
                                                                                                                                                                                                                                                                                                                                                                  "bestla::sycl_prologue_b::WeightS4T<bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T>, float>::gemv(float const*, bestla::sycl_prologue_b::ParamWeightS4<float> const&, float*, int, int, int, sycl::_V1::queue*)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",        30072,           2220827364,    48.770988,                73850,                24687,               111562
"bestla::sycl_wrapper::LauncherWOQ<bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProAT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProBTransT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::EpiT, bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T> >::compute<false>(sycl::_V1::queue*, int, int, int, int, bestla::sycl_wrapper::LauncherWOQ<bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProAT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProBTransT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::EpiT, bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T> >::Param const&)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<2>)#1}",          326,           1996897675,    43.853325,              6125452,              3729375,             13934791
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                "zeCommandListAppendMemoryCopy(M2D)",           12,            335858119,     7.375689,             27988176,                 8593,             48634427


=== Kernel Properties ===

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Kernel, Compiled, SIMD, Number of Arguments, SLM Per Work Group, Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
                                                                                                                                                                                                                                                                                                                                                                  "bestla::sycl_prologue_b::WeightS4T<bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T>, float>::gemv(float const*, bestla::sycl_prologue_b::ParamWeightS4<float> const&, float*, int, int, int, sycl::_V1::queue*)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",      JIT,   32,                   9,                  0,                         0,                       0,                           128
"bestla::sycl_wrapper::LauncherWOQ<bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProAT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProBTransT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::EpiT, bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T> >::compute<false>(sycl::_V1::queue*, int, int, int, int, bestla::sycl_wrapper::LauncherWOQ<bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProAT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::ProBTransT, bestla::sycl_ut::Benchmark_S4Fp32Fp32::EpiT, bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T> >::Param const&)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<2>)#1}",      JIT,   32,                  10,                  0,                      6144,                    3264,                           128



==================================

=== Device Timing Summary ===

                Total Execution Time (ns):          23963693866
    Total Device Time for L0 backend (ns):           6863952140

== L0 Backend ==

                                                                                                                                                                                                                    Kernel,        Calls,            Time (ns),     Time (%),         Average (ns),             Min (ns),             Max (ns)
"gemv_v2(float const*, unsigned char const*, float const*, float*, int, int, int, int, int, sycl::_V1::queue*)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",        94342,           6810988716,    99.228378,                72194,                71041,               140625
                                                                                                                                                                                      "zeCommandListAppendMemoryCopy(M2D)",           30,             52944312,     0.771339,              1764810,                 3906,              7841458
                                                                                                                                                                                      "zeCommandListAppendMemoryCopy(D2M)",           10,                19112,     0.000278,                 1911,                 1822,                 2083


=== Kernel Properties ===

                                                                                                                                                                                                                    Kernel, Compiled, SIMD, Number of Arguments, SLM Per Work Group, Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
"gemv_v2(float const*, unsigned char const*, float const*, float*, int, int, int, int, int, sycl::_V1::queue*)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",      JIT,   32,                   7,                  0,                         0,                       0,                           128



BesTLA Benchmark done

=== Device Timing Summary ===

                Total Execution Time (ns):           6494428584
    Total Device Time for L0 backend (ns):            291659037

== L0 Backend ==

                                                                                                                                                                                                                                                                                                                                                            Kernel,        Calls,            Time (ns),     Time (%),         Average (ns),             Min (ns),             Max (ns)
"bestla::sycl_prologue_b::WeightS4T<bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T>, float>::gemv(float const*, bestla::sycl_prologue_b::ParamWeightS4<float> const&, float*, int, int, int, sycl::_V1::queue*)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",         5063,            243037006,    83.329147,                48002,                44375,                76041
                                                                                                                                                                                                                                                                                                                              "zeCommandListAppendMemoryCopy(M2D)",            2,             48622031,    16.670847,             24311015,                12500,             48609531


=== Kernel Properties ===

                                                                                                                                                                                                                                                                                                                                                            Kernel, Compiled, SIMD, Number of Arguments, SLM Per Work Group, Private Memory Per Thread, Spill Memory Per Thread, Register File Size Per Thread
"bestla::sycl_prologue_b::WeightS4T<bestla::sycl_gemm::xve::GemmCoreSharedBT<bestla::sycl_gemm::xve::Config_Fp32Fp32Fp32T>, float>::gemv(float const*, bestla::sycl_prologue_b::ParamWeightS4<float> const&, float*, int, int, int, sycl::_V1::queue*)::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}",      JIT,   32,                   9,                  0,                         0,                       0,                           128
