idf_component_register(SRCS "app_main.cpp" INCLUDE_DIRS "."
    REQUIRES esp_timer esp_psram esp-tflite-micro
    EMBED_FILES "model.tflite" "golden-inputs.bin" "golden-outputs.bin")
