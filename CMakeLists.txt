cmake_minimum_required(VERSION 3.16)
project(tokenizer_c VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

add_library(tokenizer STATIC
  src/utf8.c
  src/normalize.c
  src/nfkc.c
  src/nfc.c
  src/trie.c
  src/tokenizer.c
  src/model.c
  src/segment_rules.c
  src/sentencepiece.c
  src/trainer.c
  src/tiktoken_bpe.c
)
target_include_directories(tokenizer PUBLIC include PRIVATE src)
target_link_libraries(tokenizer m)

add_executable(tok-cli cli/main.c)
target_link_libraries(tok-cli tokenizer)
if(WIN32)
  target_link_libraries(tok-cli shell32)
endif()

add_executable(test_tokenizer tests/test_tokenizer.c)
target_link_libraries(test_tokenizer tokenizer)

add_executable(test_tiktoken_bpe tests/test_tiktoken_bpe.c)
target_link_libraries(test_tiktoken_bpe tokenizer)

# CLI level regression test: `tt-decode --raw` must emit exactly the bytes
# produced by ttbpe_decode_bytes().  It spawns the built CLI binary, therefore
# the binary path is handed over at run time.
add_executable(test_cli_tt_raw tests/test_cli_tt_raw.c)

enable_testing()
add_test(NAME tokenizer_unit COMMAND test_tokenizer)
add_test(NAME tiktoken_bpe_unit COMMAND test_tiktoken_bpe)
add_test(NAME cli_tt_raw COMMAND test_cli_tt_raw $<TARGET_FILE:tok-cli>)
