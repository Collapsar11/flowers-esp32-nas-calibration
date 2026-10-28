#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cstdint>
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "esp_private/esp_clk.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "measurement_config.h"

extern const uint8_t model_data[] asm("_binary_model_tflite_start");
extern const int8_t golden_inputs[] asm("_binary_golden_inputs_bin_start");
extern const int8_t golden_inputs_end[] asm("_binary_golden_inputs_bin_end");
extern const int8_t golden_outputs[] asm("_binary_golden_outputs_bin_start");
extern const int8_t golden_outputs_end[] asm("_binary_golden_outputs_bin_end");
// Explicit static storage, accounted in the ELF, not hidden as model heap.
static int64_t durations[204];

static void wait_command(char *kind, int *session, char *nonce) {
    char line[160] = {};
    size_t pos = 0;
    for (;;) {
        const int c = getchar();
        if (c == EOF) { clearerr(stdin); vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        if (c == '\r') continue;
        if (c == '\n') {
            line[pos] = 0;
            assert(sscanf(line, "RUN %15s %d %64s", kind, session, nonce) == 3);
            assert((strcmp(kind, "timing") == 0 || strcmp(kind, "memory") == 0) && *session >= 0 && *session < 3);
            assert(strlen(nonce) == 32 && strspn(nonce, "0123456789abcdef") == 32);
            return;
        }
        assert(pos < sizeof(line)-1);
        line[pos++] = char(c);
    }
}

extern "C" void app_main(void) {
    setvbuf(stdin, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    assert(xPortGetCoreID() == 1 && esp_clk_cpu_freq() == 240000000);
    assert(esp_psram_get_size() == 8*1024*1024);
    assert(reinterpret_cast<uintptr_t>(model_data)%16 == 0);
    assert(reinterpret_cast<uintptr_t>(golden_inputs)%16 == 0);
    assert(reinterpret_cast<uintptr_t>(golden_outputs)%16 == 0);
    printf("FLOWERS_READY %s\n", MODEL_ID);
    char kind[16], nonce[65]; int session;
    wait_command(kind, &session, nonce);
    const bool timing = strcmp(kind, "timing") == 0;
    printf("FLOWERS_JSON {\"event\":\"begin\",\"id\":\"%s\",\"model_sha256\":\"%s\",\"protocol_sha256\":\"%s\",\"input_sha256\":\"%s\",\"kind\":\"%s\",\"session\":%d,\"nonce\":\"%s\",\"core\":%d,\"cpu_hz\":%d,\"psram_bytes\":%u,\"psram_mhz\":80,\"flash_mhz\":80,\"hardware_source\":\"unattested\"}\n",
        MODEL_ID, MODEL_SHA256, PROTOCOL_SHA256, INPUT_SHA256, kind, session, nonce,
        xPortGetCoreID(), esp_clk_cpu_freq(), unsigned(esp_psram_get_size()));
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
    // No probe allocation. All monitoring starts before the final arena exists.
    const size_t internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    const size_t internal_largest_before = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t psram_largest_before = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    const size_t stack_before = uxTaskGetStackHighWaterMark(nullptr);
    if (!timing) assert(heap_caps_monitor_local_minimum_free_size_start() == ESP_OK);
    constexpr size_t arena_size = ((FROZEN_ARENA_USED + 15) & ~size_t(15)) + 4096;
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    assert(arena);
    // A directly allocated heap arena must start deterministically, just as a
    // static zero-initialized tensor arena would. No earlier probe initializes it.
    memset(arena, 0, arena_size);
    {
        tflite::MicroInterpreter interpreter(model, resolver, arena, arena_size);
        assert(interpreter.AllocateTensors() == kTfLiteOk);
        assert(interpreter.arena_used_bytes() == FROZEN_ARENA_USED);
        auto *input = interpreter.input(0); auto *output = interpreter.output(0);
        assert(input->type == kTfLiteInt8 && output->type == kTfLiteInt8);
        assert(input->bytes == 160*160*3 && output->bytes == 102);
        assert(golden_inputs_end-golden_inputs == 102*input->bytes);
        assert(golden_outputs_end-golden_outputs == 102*output->bytes);
        if (timing) for (int w=0; w<10; ++w) {
            memcpy(input->data.int8, golden_inputs + input_order[session][w]*input->bytes, input->bytes);
            assert(interpreter.Invoke() == kTfLiteOk);
        }
        unsigned mismatches=0, agreement=0, maximum_error=0;
        int first_input=-1, first_output=-1, first_actual=0, first_expected=0;
        const int count = timing ? 204 : 102;
        for (int n=0; n<count; ++n) {
            const int index = input_order[session][n%102];
            memcpy(input->data.int8, golden_inputs + index*input->bytes, input->bytes);
            if (timing) {
                const int64_t before = esp_timer_get_time();
                const auto status = interpreter.Invoke();
                const int64_t after = esp_timer_get_time();
                assert(status == kTfLiteOk && after > before);
                durations[n] = after-before;
            } else assert(interpreter.Invoke() == kTfLiteOk);
            const auto *reference = golden_outputs + index*output->bytes;
            for (size_t j=0; j<output->bytes; ++j) {
                if (output->data.int8[j] != reference[j]) {
                    ++mismatches;
                    if (first_input<0) {first_input=index;first_output=j;first_actual=output->data.int8[j];first_expected=reference[j];}
                    maximum_error=std::max(maximum_error,unsigned(abs(int(output->data.int8[j])-int(reference[j]))));
                }
            }
            agreement += (std::max_element(output->data.int8, output->data.int8+output->bytes)-output->data.int8 ==
                std::max_element(reference, reference+output->bytes)-reference);
        }
        // Snapshot before any measurement logs, while the interpreter/arena are live.
        const size_t min_internal = timing ? 0 : heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        const size_t min_psram = timing ? 0 : heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM);
        const size_t internal_largest_live = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        const size_t psram_largest_live = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
        const size_t stack_after = uxTaskGetStackHighWaterMark(nullptr);
        if (!timing) assert(heap_caps_monitor_local_minimum_free_size_stop() == ESP_OK);
        if (mismatches || agreement != unsigned(count))
            printf("FLOWERS_CHECK_FAILED mismatches=%u agreement=%u count=%d max_error=%u first_input=%d output=%d actual=%d expected=%d\n",
                mismatches,agreement,count,maximum_error,first_input,first_output,first_actual,first_expected);
        assert(mismatches == 0 && agreement == unsigned(count));
        if (timing) for (int n=0; n<count; ++n)
            printf("FLOWERS_JSON {\"event\":\"invoke\",\"n\":%d,\"input\":%d,\"duration_us\":%" PRId64 "}\n", n, input_order[session][n%102], durations[n]);
        else {
            assert(min_internal <= internal_before && min_psram <= psram_before);
            printf("FLOWERS_JSON {\"event\":\"memory\",\"psram_peak_bytes\":%u,\"internal_heap_peak_bytes\":%u,\"psram_before\":%u,\"internal_before\":%u,\"psram_min_free\":%u,\"internal_min_free\":%u,\"psram_largest_before\":%u,\"internal_largest_before\":%u,\"psram_largest_live\":%u,\"internal_largest_live\":%u}\n",
                unsigned(psram_before-min_psram), unsigned(internal_before-min_internal), unsigned(psram_before), unsigned(internal_before),
                unsigned(min_psram), unsigned(min_internal), unsigned(psram_largest_before), unsigned(internal_largest_before),
                unsigned(psram_largest_live), unsigned(internal_largest_live));
        }
        printf("FLOWERS_JSON {\"event\":\"done\",\"count\":%d,\"warmup\":%d,\"integer_mismatches\":%u,\"class_agreement\":%u,\"arena_reserved\":%u,\"arena_used\":%u,\"stack_free_before_bytes\":%u,\"stack_free_after_bytes\":%u,\"stack_reserved_bytes\":16384}\n",
            count, timing ? 10 : 0, mismatches, agreement, unsigned(arena_size), unsigned(interpreter.arena_used_bytes()), unsigned(stack_before), unsigned(stack_after));
    }
    heap_caps_free(arena);
    printf("FLOWERS_MEASUREMENT_COMPLETE\n");
    for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}
