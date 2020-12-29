cmake_minimum_required(VERSION 3.10)

project(game_of_life)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
  src/main.c
  )

target_link_libraries(${PROJECT_NAME} SDL2)
