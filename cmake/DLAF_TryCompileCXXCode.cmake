function (try_compile_cxx_code code is_available)
  execute_process(
    COMMAND echo "${code}"
    COMMAND ${CMAKE_CXX_COMPILER} -c -x c++ -
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE result)

  if (NOT result)
    set(${is_available} TRUE PARENT_SCOPE)
  else()
    set(${is_available} FALSE PARENT_SCOPE)
  endif()
endfunction()
