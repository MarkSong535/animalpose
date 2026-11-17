// The C interface of the animalpose engine (`rust/src/ffi.rs`). One engine per app; every call on
// a handle comes from the main thread. Returned strings and pixel buffers are owned by the caller
// and freed with ap_string_free / ap_frame_free.

#ifndef ANIMALPOSE_H
#define ANIMALPOSE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct ApEngine ApEngine;

typedef struct {
    uint64_t sequence;
    bool overlay;
    uint32_t width;
    uint32_t height;
    uint8_t *rgba;      // NULL when the frame is the one the caller already has
    size_t len;
    int32_t animals;    // -1 without the pose overlay
    double analysis_ms;
} ApFrame;

void ap_camera_init(void);
ApEngine *ap_engine_new(void);
void ap_engine_free(ApEngine *engine);
// The engine's state as JSON, after collecting finished work.
char *ap_engine_state(ApEngine *engine);
// Runs {"cmd": ...} and returns the state, or {"error": ...}.
char *ap_engine_command(ApEngine *engine, const char *command);
bool ap_engine_frame(ApEngine *engine, uint32_t index, uint64_t known_sequence, bool known_overlay,
                     ApFrame *out);
void ap_frame_free(uint8_t *rgba, size_t len);
void ap_string_free(char *text);

#endif
