#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");
static const char *TAG = "FLOWERS_CHECK";

extern "C" void app_main(void)
{
    // Standalone inference harness, not the full bread-compact-wifi-mpu6050 application.
    ESP_LOGI(TAG, "FUNCTIONAL_VALIDATION_ONLY; NO_SILICON_LATENCY_LABELS");
    ESP_LOGI(TAG, "psram_bytes=%u", unsigned(esp_psram_get_size()));
    const auto internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const auto psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    heap_caps_monitor_local_minimum_free_size_start();
    // Fixed runtime policy across all candidates: 64 KiB activation pool on internal RAM,
    // remaining tensors in PSRAM; retain parameters in Flash to bound PSRAM consumption.
    auto *model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                               64 * 1024, dl::MEMORY_MANAGER_GREEDY, nullptr, false);
    model->profile_memory();
    ESP_ERROR_CHECK(model->test());
    const auto internal_min = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const auto psram_min = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "validation_internal_peak_delta=%u validation_psram_peak_delta=%u",
             unsigned(internal_before - internal_min), unsigned(psram_before - psram_min));
    ESP_LOGI(TAG, "GOLDEN_TEST_PASSED");
    heap_caps_monitor_local_minimum_free_size_stop();
    delete model;
    ESP_LOGI(TAG, "MODEL_DESTROYED");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
