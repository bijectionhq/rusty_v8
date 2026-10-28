// Copyright 2026 the Deno authors. All rights reserved. MIT license.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::Context;
use crate::Local;
use crate::RealIsolate;
use crate::scope::{GetIsolate, PinScope};

#[repr(C)]
struct NativeBoundedStackSampler {
  _private: [u8; 0],
}

unsafe extern "C" {
  fn v8__BoundedStackSampler__New(
    isolate: *mut RealIsolate,
    context: *const Context,
    options: *const BoundedStackSamplingOptions,
  ) -> *mut NativeBoundedStackSampler;
  fn v8__BoundedStackSampler__Stop(sampler: *mut NativeBoundedStackSampler);
  fn v8__BoundedStackSampler__Summary(
    sampler: *const NativeBoundedStackSampler,
    summary: *mut BoundedStackSamplingSummary,
  );
  fn v8__BoundedStackSampler__Sample(
    sampler: *const NativeBoundedStackSampler,
    index: u32,
    sample: *mut BoundedStackSample,
  ) -> bool;
  fn v8__BoundedStackSampler__Dispose(sampler: *mut NativeBoundedStackSampler);
}

/// Limits may only reduce the native fixed capacities. Intervals are target
/// intervals in 1000..=10000 microseconds. Collection cooperatively stops after
/// one second, 65,536 code events or any configured capacity. Time is checked
/// at rundown/code-event/timer boundaries; heap setup, safepoints and scheduling
/// can exceed it. The collector holds its context weakly.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BoundedStackSamplingOptions {
  pub sampling_interval_us: u32,
  pub max_samples: u32,
  pub max_code_entries: u32,
  pub max_rundown_objects: u32,
}

impl Default for BoundedStackSamplingOptions {
  fn default() -> Self {
    Self {
      sampling_interval_us: 1000,
      max_samples: 256,
      max_code_entries: 4096,
      max_rundown_objects: 65536,
    }
  }
}

/// An unmapped frame has both fields -1. This is the physical compiled
/// function container's start offset in UTF16 code units, not a sampled
/// instruction/source position. Inline, native and Wasm frames are absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BoundedStackFrame {
  pub script_id: i32,
  pub function_start_utf16: i32,
}

bitflags! {
  /// Missing detail and collection cutoffs. Scheduling jitter and coalesced or
  /// undelivered signals have unknown counts and are not represented as zero.
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub struct BoundedStackSamplingLoss: u32 {
    const SAMPLE_LIMIT = 1 << 0;
    const CODE_LIMIT = 1 << 1;
    const RUNDOWN_LIMIT = 1 << 2;
    const DEPTH_LIMIT = 1 << 3;
    const UNMAPPED_FRAME = 1 << 4;
    const INTERRUPTED_MAP_MUTATION = 1 << 5;
    const STACK_UNAVAILABLE = 1 << 6;
    const CONCURRENT_CODE_CREATION = 1 << 7;
    const RELOCATION_CONFLICT = 1 << 8;
    const CODE_EVENT_LIMIT = 1 << 9;
    const TIME_LIMIT = 1 << 10;
    const CONTEXT_COLLECTED = 1 << 11;
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BoundedStackSample {
  pub offset_us: i64,
  frame_count: u32,
  loss: u32,
  frames: [BoundedStackFrame; 32],
}

impl BoundedStackSample {
  /// Topmost physical frame first.
  pub fn frames(&self) -> &[BoundedStackFrame] {
    &self.frames[..self.frame_count as usize]
  }
  pub fn loss(&self) -> BoundedStackSamplingLoss {
    BoundedStackSamplingLoss::from_bits_retain(self.loss)
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BoundedStackSamplingSummary {
  pub sample_count: u32,
  loss: u32,
  pub non_js_observations: u32,
  pub other_context_observations: u32,
  pub map_mutation_observations: u32,
  pub unavailable_stack_observations: u32,
  pub elapsed_us: i64,
  pub collection_end_us: i64,
}

impl BoundedStackSamplingSummary {
  pub fn loss(&self) -> BoundedStackSamplingLoss {
    BoundedStackSamplingLoss::from_bits_retain(self.loss)
  }
}

/// Owned numeric observations. No names, input values, script URLs, code
/// addresses or V8 handles escape the collector. Samples measure observation
/// frequency, not exact processor CPU durations.
#[derive(Debug)]
pub struct BoundedStackSamplingReport {
  pub summary: BoundedStackSamplingSummary,
  pub samples: Vec<BoundedStackSample>,
}

/// A context-scoped native sampler. Drop always stops and disposes it, including
/// unwinding and terminated execution. It is neither Send nor Sync. Forgetting
/// the guard can leak its bounded report until isolate teardown, which also
/// stops/disposes any outstanding native collector.
///
/// The guard cannot escape its handle scope:
/// ```compile_fail
/// fn escape<'s>(isolate: &mut v8::OwnedIsolate) -> v8::BoundedStackSampler<'s> {
///   v8::scope!(let scope, isolate);
///   let context = v8::Context::new(scope, Default::default());
///   let scope = &mut v8::ContextScope::new(scope, context);
///   v8::BoundedStackSampler::new(scope, Default::default()).unwrap()
/// }
/// ```
/// It cannot be sent to a sampling or cleanup thread:
/// ```compile_fail
/// fn requires_send<T: Send>() {}
/// requires_send::<v8::BoundedStackSampler<'static>>();
/// ```
pub struct BoundedStackSampler<'s> {
  native: NonNull<NativeBoundedStackSampler>,
  _context: Local<'s, Context>,
  _not_send_sync: PhantomData<Rc<()>>,
}

impl<'s> BoundedStackSampler<'s> {
  /// Starts sampling the scope's current context. None means invalid bounds,
  /// collector capacity or unsupported platform/thread/sandbox coverage.
  /// Ordinary execution should continue without profiling.
  pub fn new(
    scope: &PinScope<'s, '_>,
    options: BoundedStackSamplingOptions,
  ) -> Option<Self> {
    let context = scope.get_current_context();
    let native = NonNull::new(unsafe {
      v8__BoundedStackSampler__New(scope.get_isolate_ptr(), &*context, &options)
    })?;
    Some(Self {
      native,
      _context: context,
      _not_send_sync: PhantomData,
    })
  }

  /// Stops sampling and copies the bounded numeric result. Consuming the guard
  /// releases native admission before the caller starts another collector.
  pub fn finish(self) -> BoundedStackSamplingReport {
    unsafe { v8__BoundedStackSampler__Stop(self.native.as_ptr()) };
    let mut summary = MaybeUninit::uninit();
    unsafe {
      v8__BoundedStackSampler__Summary(
        self.native.as_ptr(),
        summary.as_mut_ptr(),
      );
    }
    let summary = unsafe { summary.assume_init() };
    let mut samples = Vec::with_capacity(summary.sample_count as usize);
    for index in 0..summary.sample_count {
      let mut sample = MaybeUninit::uninit();
      let present = unsafe {
        v8__BoundedStackSampler__Sample(
          self.native.as_ptr(),
          index,
          sample.as_mut_ptr(),
        )
      };
      assert!(
        present,
        "native bounded sampler returned an invalid sample index"
      );
      samples.push(unsafe { sample.assume_init() });
    }
    BoundedStackSamplingReport { summary, samples }
  }
}

impl Drop for BoundedStackSampler<'_> {
  fn drop(&mut self) {
    unsafe { v8__BoundedStackSampler__Dispose(self.native.as_ptr()) };
  }
}
