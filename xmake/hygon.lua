local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
toolchain("hygon.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    -- 使用DTK中的CUDA编译器
    local nvcc_path = path.join(dtk_root, "cuda", "bin", "nvcc")
    if os.isfile(nvcc_path) then
        set_toolset("cu"  , nvcc_path)
        set_toolset("culd", nvcc_path)
    else
        set_toolset("cu"  , "nvcc")
        set_toolset("culd", "nvcc")
    end
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
toolchain_end()

rule("hygon.env")
    -- Fix the deprecated warning by using add_orders
    add_orders("cuda.env", "hygon.env")
    after_load(function (target)
        -- This logic to remove CUDA-specific libs is correct and can remain
        local old = target:get("syslinks") or {}
        local new = {}
        for _, link in ipairs(old) do
            if link ~= "cudadevrt" and link ~= "cudnn" then
                table.insert(new, link)
            end
        end
        if #old > #new then
            target:set("syslinks", new)
            print("CUDA specific libraries removed for Hygon DCU. New syslinks: {" .. table.concat(new, ", ") .. "}")
        end
    end)
rule_end()

local function resolve_hygon_arch()
    local configured = get_config("hygon-arch")
    if configured and configured ~= "" then
        return configured
    end

    local env_arch = os.getenv("HYGON_ARCH")
    if env_arch and env_arch ~= "" then
        return env_arch
    end

    return "gfx936"
end

local HYGON_ARCH = resolve_hygon_arch()
print("编译海光DCU架构: " .. HYGON_ARCH)

target("infiniop-hygon")
    set_kind("static")
    set_languages("cxx17")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("hygon.toolchain")
    add_rules("hygon.env")
    set_values("cuda.rdc", false)

    -- 海光DCU使用DTK中的CUDA库
    add_links("cudart", "cublas", "curand", "cublasLt", "cudnn")
    
    -- 添加DTK路径支持
    local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
    if os.isdir(dtk_root) then
        add_includedirs(path.join(dtk_root, "include"))
        add_includedirs(path.join(dtk_root, "cuda", "include"))
        add_includedirs(path.join(dtk_root, "cuda", "cuda-11", "extras", "clang_internal_header"))
        add_linkdirs(path.join(dtk_root, "lib"))
        add_linkdirs(path.join(dtk_root, "cuda", "lib64"))
    end

    set_warnings("all", "error")
    add_cuflags("-Wno-error=unused-private-field")
    add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
    add_cuflags("-Wno-error=macro-redefined", {force = true})
    add_cuflags("-Wno-error=ignored-attributes", {force = true})
    add_cuflags("-Wno-error=uninitialized", {force = true})
    add_cuflags("-Wno-error=unused-variable", {force = true})
    add_cuflags("-Wno-error=unused-function", {force = true})
    add_cuflags("-Wno-error=int-to-void-pointer-cast", {force = true})
    add_cuflags("-Xclang", "-fno-cuda-host-device-constexpr", {force = true})
    add_cuflags("-fPIC", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")
    add_cxxflags("-fPIC")

    add_cuflags("-arch=" .. HYGON_ARCH)
    
    -- Keep CPU descriptors available because ENABLE_CPU_API is enabled globally.
    add_files("../src/infiniop/devices/cpu/*.cc", "../src/infiniop/ops/*/cpu/*.cc", "../src/infiniop/reduce/cpu/*.cc")

    -- 复用NVIDIA的CUDA实现，通过HIP兼容层
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu")

    -- Keep platform-specific or currently unregistered NVIDIA sources out of the Hygon target.
    remove_files("../src/infiniop/ops/avg_pool3d/nvidia/*.cu")
    remove_files("../src/infiniop/ops/dequant*/nvidia/*.cu")
    remove_files("../src/infiniop/ops/dot/nvidia/*.cu")
    remove_files("../src/infiniop/ops/dist/nvidia/*.cu")
    remove_files("../src/infiniop/ops/gptq_qyblas_gemm/nvidia/*.cu")
    remove_files("../src/infiniop/ops/histc/nvidia/*.cu")
    remove_files("../src/infiniop/ops/quant*/nvidia/*.cu")
    remove_files("../src/infiniop/ops/scaled_mm/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", "../build/ninetoothed/*.cpp", {cxxflags = {"-Wno-return-type"}})
    end
target_end()

target("infinirt-hygon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("hygon.toolchain")
    add_rules("hygon.env")
    set_values("cuda.rdc", false)

    add_links("cudart", "curand")
    
    -- 添加DTK路径支持
    local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
    if os.isdir(dtk_root) then
        add_includedirs(path.join(dtk_root, "include"))
        add_includedirs(path.join(dtk_root, "cuda", "include"))
        add_includedirs(path.join(dtk_root, "cuda", "cuda-11", "extras", "clang_internal_header"))
        add_linkdirs(path.join(dtk_root, "lib"))
        add_linkdirs(path.join(dtk_root, "cuda", "lib64"))
    end

    set_warnings("all", "error")
    add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
    add_cuflags("-Wno-error=macro-redefined", {force = true})
    add_cuflags("-Wno-error=ignored-attributes", {force = true})
    add_cuflags("-Wno-error=uninitialized", {force = true})
    add_cuflags("-Wno-error=unused-variable", {force = true})
    add_cuflags("-Wno-error=unused-function", {force = true})
    add_cuflags("-Wno-error=int-to-void-pointer-cast", {force = true})
    add_cuflags("-Xclang", "-fno-cuda-host-device-constexpr", {force = true})
    add_cuflags("-fPIC", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")
    add_cxxflags("-fPIC")

    add_cuflags("-arch=" .. HYGON_ARCH)
    
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-hygon")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)

    if has_config("ccl") then
        set_toolchains("hygon.toolchain")
        add_rules("hygon.env")
        set_values("cuda.rdc", false)

        add_links("cudart", "curand")
        
        -- 添加DTK路径支持
        local dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
        if os.isdir(dtk_root) then
            add_includedirs(path.join(dtk_root, "include"))
            add_includedirs(path.join(dtk_root, "cuda", "include"))
            add_includedirs(path.join(dtk_root, "cuda", "cuda-11", "extras", "clang_internal_header"))
            add_linkdirs(path.join(dtk_root, "lib"))
            add_linkdirs(path.join(dtk_root, "cuda", "lib64"))
        end

        set_warnings("all", "error")
        add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
        add_cuflags("-Wno-error=macro-redefined", {force = true})
        add_cuflags("-Wno-error=ignored-attributes", {force = true})
        add_cuflags("-Wno-error=uninitialized", {force = true})
        add_cuflags("-Wno-error=unused-variable", {force = true})
        add_cuflags("-Wno-error=unused-function", {force = true})
        add_cuflags("-Wno-error=int-to-void-pointer-cast", {force = true})
        add_cuflags("-Xclang", "-fno-cuda-host-device-constexpr", {force = true})
        add_cuflags("-fPIC", "-std=c++17", {force = true})
        add_culdflags("-fPIC")
        add_cxflags("-fPIC")
        add_cxxflags("-fPIC")

        -- 添加海光DCU特定的编译标志
        -- 检测实际GPU架构，如果未指定则默认使用gfx906
        local hygon_arch = os.getenv("HYGON_ARCH") or "gfx906"
        add_cuflags("-arch=" .. hygon_arch)

        -- 使用NCCL (NVIDIA Collective Communications Library)
        add_links("nccl")

        add_files("../src/infiniccl/cuda/*.cu")
    end
target_end()

local FLASH_ATTN_ROOT = get_config("flash-attn")

local function hygon_flash_attn_cuda_so_path()
    local env_path = os.getenv("FLASH_ATTN_2_CUDA_SO")
    if env_path and env_path ~= "" then
        env_path = env_path:trim()
        if os.isfile(env_path) then
            return env_path
        end
        print(string.format("warning: hygon+flash-attn: FLASH_ATTN_2_CUDA_SO is not a file: %s, fallback to container/default path", env_path))
    end

    local container_path = os.getenv("FLASH_ATTN_HYGON_CUDA_SO_CONTAINER")
    if not container_path or container_path == "" then
        container_path = "/usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"
    end

    if not os.isfile(container_path) then
        print(
            string.format(
                "warning: hygon+flash-attn: expected %s; install flash-attn in the active Python env, or export FLASH_ATTN_2_CUDA_SO.",
                container_path
            )
        )
    end
    return container_path
end

target("flash-attn-hygon")
    set_kind("phony")
    set_default(false)

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_build(function (target)
            local TORCH_DIR = os.iorunv("python", {"-c", "import torch, os; print(os.path.dirname(torch.__file__))"}):trim()
            local PYTHON_INCLUDE = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim()
            local PYTHON_LIB_DIR = os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"}):trim()

            target:add("includedirs", TORCH_DIR .. "/include", TORCH_DIR .. "/include/torch/csrc/api/include", PYTHON_INCLUDE, {public = false})
            target:add("linkdirs", TORCH_DIR .. "/lib", PYTHON_LIB_DIR, {public = false})
        end)
    else
        before_build(function (target)
            print("Flash Attention not available, skipping flash-attn-hygon integration")
        end)
    end
target_end()

target("infinicore_cpp_api")
    add_defines("__HIP_PLATFORM_AMD__")
    add_defines("C10_CUDA_NO_CMAKE_CONFIGURE_FILE")
    add_defines("TORCH_CUDA_CPP_API=TORCH_HIP_CPP_API")

    if has_config("aten") then
        add_defines("ENABLE_ATEN")
        if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
            add_defines("ENABLE_FLASH_ATTN")
            add_packages("pybind11")
        end
    end

    local dtk_root_cppapi = os.getenv("DTK_ROOT") or "/opt/dtk"
    if os.isdir(dtk_root_cppapi) then
        add_includedirs(
            path.join(dtk_root_cppapi, "include"),
            path.join(dtk_root_cppapi, "cuda", "include"),
            path.join(dtk_root_cppapi, "cuda", "cuda", "include"),
            {public = true}
        )
        add_linkdirs(
            path.join(dtk_root_cppapi, "lib"),
            path.join(dtk_root_cppapi, "cuda", "lib64"),
            path.join(dtk_root_cppapi, "cuda", "cuda", "lib64"),
            {public = true}
        )
    end

    if FLASH_ATTN_ROOT and FLASH_ATTN_ROOT ~= "" then
        before_link(function (target)
            local flash_so_hygon = hygon_flash_attn_cuda_so_path()
            local flash_dir_hygon = path.directory(flash_so_hygon)
            local flash_name_hygon = path.filename(flash_so_hygon)
            local flash_lib_dir_hygon = path.join(FLASH_ATTN_ROOT, "flash_attn", "lib")
            target:add(
                "shflags",
                "-Wl,--no-as-needed -L" .. flash_lib_dir_hygon .. " -l:libflash_attention.so -Wl,-rpath," .. flash_lib_dir_hygon,
                "-L" .. flash_dir_hygon .. " -l:" .. flash_name_hygon .. " -Wl,-rpath," .. flash_dir_hygon,
                {force = true}
            )
        end)
    end
target_end()
