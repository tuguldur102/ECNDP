cmake_minimum_required(VERSION 3.16)
project(ECNDP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ortools REQUIRED)

add_executable(ecndp
  algorithms/main.cpp
  algorithms/utils.cpp
  algorithms/exact.cpp
  algorithms/local_search.cpp
  algorithms/improvement_condition.cpp
  algorithms/greedy_empty_set.cpp
  algorithms/greedy_mis.cpp
)

target_include_directories(ecndp PRIVATE c_env/algorithms)
target_link_libraries(ecndp PRIVATE ortools::ortools)