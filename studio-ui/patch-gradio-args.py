"""Realign the UI's positional /generation_wrapper args with ACE-Step v0.1.8's
72-parameter Gradio signature. The pinned UI commit predates the params added
since (sampler mode, DCW, MP3 bitrate/rate, fade in/out, repaint mode, ...), so
everything from index 20 on was shifted and Gradio rejected the call. Indices
0-19 are unchanged. Fails the build if the anchor text is not found, so a UI
bump can't silently skip the patch."""
import re, sys
p = "/app/server/src/services/acestep.ts"
s = open(p).read()
# start at the old Task Type slot (index 20): the new signature has no task-type
# param (it is derived from source audio / instruction), so that slot becomes no_fsq.
start = s.index("    (params.taskType === 'audio2audio'")
end_marker = "    params.autogen ?? false,"
end = s.index(end_marker, start) + len(end_marker)
new = """    false,                                                        // 20: no_fsq
    params.useAdg ?? false,                                       // 21: Use ADG
    params.cfgIntervalStart ?? 0.0,                               // 22: CFG Interval Start
    params.cfgIntervalEnd ?? 1.0,                                 // 23: CFG Interval End
    params.shift ?? 3.0,                                          // 24: Shift
    params.inferMethod || 'ode',                                  // 25: Inference Method
    'euler',                                                      // 26: Sampler Mode
    0.0,                                                          // 27: Velocity Norm Threshold
    0.0,                                                          // 28: Velocity EMA Factor
    !(process as any).getBuiltinModule('fs').existsSync('/flags/dcw_off'), // 29: Enable DCW (off while /flags/dcw_off exists; music-api sets it for sft)
    'double',                                                     // 30: DCW Mode
    0.05,                                                         // 31: DCW Scaler
    0.02,                                                         // 32: DCW High Scaler
    'haar',                                                       // 33: DCW Wavelet
    params.customTimesteps || '',                                 // 34: Custom Timesteps
    params.audioFormat || 'mp3',                                  // 35: Audio Format
    '192k',                                                       // 36: MP3 Bitrate
    48000,                                                        // 37: MP3 Sample Rate
    params.lmTemperature ?? 0.85,                                 // 38: LM Temperature
    isThinking,                                                   // 39: Think
    params.lmCfgScale ?? 2.0,                                     // 40: LM CFG Scale
    params.lmTopK ?? 0,                                           // 41: LM Top-K
    params.lmTopP ?? 0.9,                                         // 42: LM Top-P
    params.lmNegativePrompt || 'NO USER INPUT',                   // 43: LM Negative Prompt
    useCot ? (params.useCotMetas ?? true) : false,                // 44: CoT Metas
    useCot ? (params.useCotCaption ?? true) : false,              // 45: CaptionRewrite
    useCot ? (params.useCotLanguage ?? true) : false,             // 46: CoT Language
    params.constrainedDecodingDebug ?? false,                     // 47: Constrained Decoding Debug
    params.allowLmBatch ?? true,                                  // 48: ParallelThinking
    params.getScores ?? false,                                    // 49: Auto Score
    params.getLrc ?? false,                                       // 50: Auto LRC
    params.scoreScale ?? 0.5,                                     // 51: Quality Score Sensitivity
    params.lmBatchChunkSize ?? 8,                                 // 52: LM Batch Chunk Size
    params.trackName || null,                                     // 53: Track Name
    params.completeTrackClasses || [],                            // 54: Track Names
    true,                                                         // 55: Enable Normalization
    -1.0,                                                         // 56: Target Peak (dB)
    0.0,                                                          // 57: Fade In
    0.0,                                                          // 58: Fade Out
    0.0,                                                          // 59: Latent Shift
    1.0,                                                          // 60: Latent Rescale
    'balanced',                                                   // 61: Repaint Mode
    0.5,                                                          // 62: Repaint Strength
    0.0,                                                          // 63: variance
    '',                                                           // 64: seed (retake)
    false,                                                        // 65: Edit
    '',                                                           // 66: source caption
    '',                                                           // 67: source lyrics
    0.0,                                                          // 68: n_min
    1.0,                                                          // 69: n_max
    1,                                                            // 70: n_avg
    params.autogen ?? false,                                      // 71: AutoGen"""
s = s[:start] + new + s[end:]
open(p, "w").write(s)
print("patched", p)
