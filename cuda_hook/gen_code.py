import subprocess
import re

LIB_CUDA_SO = '/usr/lib/x86_64-linux-gnu/libcuda.so'
CUDA_HEADER = '/usr/local/cuda/include/cuda.h'

def get_all_func():
  try:
    cmd = "nm -D {SO} | grep -i cu | awk '{{print $3}}'".format(SO=LIB_CUDA_SO)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    funcs = result.stdout
  except subprocess.CalledProcessError as e:
    print(f"Error executing nm command: {e}")
    exit(-1)
  return funcs.strip().split('\n')

#
# input : `CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr);`
# return: (CUresult, const char **), (error, pStr), [CUresult error, const char **pStr]
#
def get_sig_for(func):
  cmd = f"sed -n ':start; /;$/!{{N; b start}}; /CUresult CUDAAPI.*{func}(/ {{p; /);/q}}' {CUDA_HEADER} | grep -A 20 'CUresult CUDAAPI' || true"
  result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
  sig = result.stdout.strip()
  param_string_match = re.search(r'\((.*)\)', sig, re.DOTALL)
  if not param_string_match:
    # print(f'{func} sig not found, skip.')
    return None
  param_string = param_string_match.group(1)
  raw_params = [param.strip() for param in param_string.split(',')]
  param_types = []
  param_names = []
  for param in raw_params:
    if param == 'void':
      param_types.append('void')
      param_names.append('')
    else:
      param_pattern = r'(.+?)(\w+)$'
      param_match = re.search(param_pattern, param)
      if param_match:
        raw_type = param_match.group(1).strip()
        variable = param_match.group(2).strip()
        param_types.append(raw_type)
        param_names.append(variable)
      else:
        print(f"parse params {param} failed")
        exit(-1)
  return param_types, param_names, raw_params


def gen_cpp_code_for_func(func):
  template = r"""
HOOK_C_API HOOK_DECL_EXPORT CUresult {FUNC_NAME}({FUNC_ARGS}) {{
  using func_ptr = CUresult (*)({FUNC_TYPES});
  static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("{FUNC_NAME}"));
  return func_entry({FUNC_ARGUMENTS});
}}
"""
  sig_func = func
  param_info = get_sig_for(sig_func)
  if param_info is None:
    # print(f"{sig_func} not found a sig, try remove '_v2' to match sig")
    sig_func = sig_func.replace('_v2', '')
    param_info = get_sig_for(sig_func)
  
  if param_info is None:
    # print(f"{sig_func} not found a sig, try remove '_ptsz' to match sig")
    sig_func = sig_func.replace('_ptsz', '')
    param_info = get_sig_for(sig_func)

  if param_info is None:
    # print(f"{sig_func} not found a sig, try remove '_ptds' to match sig")
    sig_func = sig_func.replace('_ptds', '')
    param_info = get_sig_for(sig_func)

  if param_info is None:
    print(f"{func} cpp code not generate. Not found sig in header file.")
    return None

  (param_types, param_names, raw_params) = param_info
  return template.format(FUNC_NAME=func, FUNC_ARGS=','.join(raw_params), FUNC_TYPES=','.join(param_types), FUNC_ARGUMENTS=','.join(param_names))


def main():
  funcs = get_all_func()

  cpp_code = r'''
#include <cuda.h>
#include "cuda_hook.h"

#undef cuDeviceTotalMem
#undef cuCtxCreate
#undef cuCtxCreate_v3
#undef cuModuleGetGlobal
#undef cuMemGetInfo
#undef cuMemAlloc
#undef cuMemAllocPitch
#undef cuMemFree
#undef cuMemGetAddressRange
#undef cuMemAllocHost
#undef cuMemHostGetDevicePointer
#undef cuMemcpyHtoD
#undef cuMemcpyDtoH
#undef cuMemcpyDtoD
#undef cuMemcpyDtoA
#undef cuMemcpyAtoD
#undef cuMemcpyHtoA
#undef cuMemcpyAtoH
#undef cuMemcpyAtoA
#undef cuMemcpyHtoAAsync
#undef cuMemcpyAtoHAsync
#undef cuMemcpy2D
#undef cuMemcpy2DUnaligned
#undef cuMemcpy3D
#undef cuMemcpyHtoDAsync
#undef cuMemcpyDtoHAsync
#undef cuMemcpyDtoDAsync
#undef cuMemcpy2DAsync
#undef cuMemcpy3DAsync
#undef cuMemsetD8
#undef cuMemsetD16
#undef cuMemsetD32
#undef cuMemsetD2D8
#undef cuMemsetD2D16
#undef cuMemsetD2D32
#undef cuArrayCreate
#undef cuArrayGetDescriptor
#undef cuArray3DCreate
#undef cuArray3DGetDescriptor
#undef cuTexRefSetAddress
#undef cuTexRefGetAddress
#undef cuGraphicsResourceGetMappedPointer
#undef cuCtxDestroy
#undef cuCtxPopCurrent
#undef cuCtxPushCurrent
#undef cuStreamDestroy
#undef cuEventDestroy
#undef cuTexRefSetAddress2D
#undef cuLinkCreate
#undef cuLinkAddData
#undef cuLinkAddFile
#undef cuMemHostRegister
#undef cuGraphicsResourceSetMapFlags
#undef cuStreamBeginCapture
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxSetFlags
#undef cuDeviceGetUuid_v2
#undef cuIpcOpenMemHandle
#undef cuGraphInstantiate

#define HOOK_C_API extern "C"
#define HOOK_DECL_EXPORT __attribute__((visibility("default")))

constexpr const char* LIB_CUDA_SO{"/usr/lib/x86_64-linux-gnu/libcuda.so"};

class HookSingleton {
public:
  static HookSingleton& GetInstance() {
    static auto* single = new HookSingleton();
    return *single;
  }

  void* GetSymbol(const char* name) { return lib_.SymbolAddress(name); }

private:
  HookSingleton() : lib_(LIB_CUDA_SO) {}
  DynamicLibrary lib_;
};

#define HOOK_CUDA_SYMBOL(func_name) HookSingleton::GetInstance().GetSymbol(func_name)
'''
  for func in funcs:
    code = gen_cpp_code_for_func(func)
    if not code:
      continue
    cpp_code += code
    cpp_code += '\n'

  with open('cuda_hook.gen.cc', 'w') as f:
    f.write(cpp_code)

if __name__ == '__main__':
  main()
