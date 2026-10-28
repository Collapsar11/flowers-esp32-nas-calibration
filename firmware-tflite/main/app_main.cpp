#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const uint8_t model_data[] asm("_binary_model_tflite_start");
extern const int8_t test_inputs[] asm("_binary_golden_inputs_bin_start");
extern const int8_t test_inputs_end[] asm("_binary_golden_inputs_bin_end");
extern const int8_t test_outputs[] asm("_binary_golden_outputs_bin_start");
extern const int8_t test_outputs_end[] asm("_binary_golden_outputs_bin_end");
static const char *TAG = "FLOWERS_CHECK";

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "FUNCTIONAL_VALIDATION_ONLY; NO_SILICON_LATENCY_LABELS");
    ESP_LOGI(TAG, "psram_bytes=%u", unsigned(esp_psram_get_size()));
    const auto *model = tflite::GetModel(model_data);
    assert(model->version() == TFLITE_SCHEMA_VERSION);
    tflite::MicroMutableOpResolver<12> resolver;
    assert(resolver.AddConv2D() == kTfLiteOk);
    assert(resolver.AddDepthwiseConv2D() == kTfLiteOk);
    assert(resolver.AddFullyConnected() == kTfLiteOk);
    assert(resolver.AddAdd() == kTfLiteOk);
    assert(resolver.AddMul() == kTfLiteOk);
    assert(resolver.AddHardSwish() == kTfLiteOk);
    assert(resolver.AddMean() == kTfLiteOk);
    assert(resolver.AddReshape() == kTfLiteOk);
    assert(resolver.AddPad() == kTfLiteOk);
    assert(resolver.AddRelu6() == kTfLiteOk);
    assert(resolver.AddAveragePool2D() == kTfLiteOk);
    assert(resolver.AddQuantize() == kTfLiteOk);
    // Probe the runtime's actual arena requirement, then reserve only that plus
    // fixed alignment headroom. A fixed 6 MiB allocation is not a model RAM metric.
    constexpr size_t probe_size = 6 * 1024 * 1024 - 4096;
    auto *probe = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, probe_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    assert(probe);
    size_t arena_size;
    {
        tflite::MicroInterpreter probe_interpreter(model, resolver, probe, probe_size);
        assert(probe_interpreter.AllocateTensors() == kTfLiteOk);
        arena_size = ((probe_interpreter.arena_used_bytes() + 15) & ~size_t(15)) + 4096;
    }
    heap_caps_free(probe);
    assert(arena_size <= probe_size);
    const auto before_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const auto before_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    heap_caps_monitor_local_minimum_free_size_start();
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    assert(arena);
    {
        tflite::MicroInterpreter interpreter(model, resolver, arena, arena_size);
        assert(interpreter.AllocateTensors() == kTfLiteOk);
        auto *input = interpreter.input(0);
        auto *output = interpreter.output(0);
        assert(input->type == kTfLiteInt8 && output->type == kTfLiteInt8);
        const size_t count = (test_inputs_end - test_inputs) / input->bytes;
        assert(count > 0 && count * output->bytes == size_t(test_outputs_end - test_outputs));
        ESP_LOGI(TAG, "tensor_arena_reserved_bytes=%u tensor_arena_used_bytes=%u", unsigned(arena_size), unsigned(interpreter.arena_used_bytes()));
        unsigned mismatches = 0, class_agreement = 0, maximum_error = 0;
        for (size_t n = 0; n < count; ++n) {
            memcpy(input->data.int8, test_inputs + n * input->bytes, input->bytes);
            assert(interpreter.Invoke() == kTfLiteOk);
            const int8_t *expected = test_outputs + n * output->bytes;
            for (size_t j = 0; j < output->bytes; ++j) {
                const unsigned delta = abs(int(output->data.int8[j]) - int(expected[j]));
                mismatches += delta != 0;
                maximum_error = std::max(maximum_error, delta);
            }
            const auto predicted = std::max_element(output->data.int8, output->data.int8 + output->bytes) - output->data.int8;
            const auto reference = std::max_element(expected, expected + output->bytes) - expected;
            class_agreement += predicted == reference;
            ESP_LOGI(TAG, "sample=%u class=%u reference=%u", unsigned(n), unsigned(predicted), unsigned(reference));
        }
        ESP_LOGI(TAG, "golden_count=%u class_agreement=%u integer_mismatches=%u max_integer_error=%u", unsigned(count), class_agreement, mismatches, maximum_error);
        ESP_LOGI(TAG, "validation_internal_peak_delta=%u validation_psram_peak_delta=%u",
            unsigned(before_internal - heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
            unsigned(before_psram - heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM)));
        if (mismatches == 0) ESP_LOGI(TAG, "GOLDEN_TEST_PASSED");
        else ESP_LOGE(TAG, "GOLDEN_TEST_FAILED");
    }
    heap_caps_monitor_local_minimum_free_size_stop();
    heap_caps_free(arena);
    ESP_LOGI(TAG, "MODEL_DESTROYED");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
