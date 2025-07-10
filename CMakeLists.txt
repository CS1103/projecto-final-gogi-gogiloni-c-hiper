cmake_minimum_required(VERSION 3.18)
project(proyecto_final_nn)

set(CMAKE_CXX_STANDARD 17)

# Ejecutable para entrenamiento
add_executable(proyecto_final_train
    src/main.cpp
)

# Ejecutable para pruebas
add_executable(proyecto_final_test
    src/main2.cpp
)

target_include_directories(proyecto_final_train PRIVATE
    src
)

target_include_directories(proyecto_final_test PRIVATE
    src
)
