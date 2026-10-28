# ESP-IDF's generic EMBED_FILES assembly does not align each binary symbol.
# FlatBuffer scalar fields and optimized kernels require aligned model data.
if(NOT CMAKE_BUILD_EARLY_EXPANSION)
    configure_file(embedded.S.in ${CMAKE_CURRENT_BINARY_DIR}/measurement_embedded.S @ONLY)
endif()
idf_component_register(SRCS "app_main.cpp" "${CMAKE_CURRENT_BINARY_DIR}/measurement_embedded.S" INCLUDE_DIRS "."
    REQUIRES esp_timer esp_psram esp-tflite-micro esp_hw_support)
if(NOT CMAKE_BUILD_EARLY_EXPANSION)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/measurement_embedded.S PROPERTIES
        OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/model.tflite;${CMAKE_CURRENT_SOURCE_DIR}/golden-inputs.bin;${CMAKE_CURRENT_SOURCE_DIR}/golden-outputs.bin")
endif()
