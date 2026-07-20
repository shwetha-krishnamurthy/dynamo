// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::common::handoff::HandoffId;
use crate::common::protocols::ForwardPassSnapshot;
use crate::scheduler::SchedulerCommandResult;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

struct FakeCore {
    publishers: KvEventPublishers,
    command_effects: bool,
    midpass_kv_effects: bool,
    metrics: MockerMetrics,
    pass_duration: Duration,
    execute_count: Option<Arc<AtomicUsize>>,
    cancel_after_execute: Option<(usize, CancellationToken)>,
}

impl FakeCore {
    fn publish_kv(&self, event_id: u64) {
        self.publishers
            .publish(
                KvCacheEvent {
                    event_id,
                    data: KvCacheEventData::Cleared,
                    dp_rank: 0,
                },
                None,
            )
            .unwrap();
    }
}

impl LiveBoundaryCore for FakeCore {
    fn live_is_empty(&self) -> bool {
        self.metrics.running_requests == 0 && self.metrics.waiting_requests == 0
    }

    fn receive_live_request(&mut self, _request: crate::common::protocols::DirectRequest) {
        self.metrics.waiting_requests += 1;
    }

    fn apply_live_command(
        &mut self,
        command: SchedulerCommand,
        _allow_destination_admission: bool,
        _now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        let cancel_request = matches!(&command, SchedulerCommand::CancelRequest { .. });
        let result = if cancel_request && self.metrics.running_requests == 0 {
            SchedulerCommandResult::Noop
        } else {
            SchedulerCommandResult::Applied
        };
        if cancel_request {
            self.metrics.running_requests = self.metrics.running_requests.saturating_sub(1);
        }
        let mut effects = SchedulerCommandEffects::new(result);
        if self.command_effects || self.midpass_kv_effects {
            self.publish_kv(2);
        }
        if self.command_effects {
            effects
                .lifecycle_events
                .push(SchedulerLifecycleEvent::DestinationReserved {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                    request_id: Uuid::from_u128(3),
                    transferable_prompt_tokens: 4,
                });
        }
        Ok(effects)
    }

    fn retry_live_destinations(&mut self, _now_ms: f64) -> Vec<SchedulerLifecycleEvent> {
        Vec::new()
    }

    fn live_metrics(&self) -> MockerMetrics {
        self.metrics.clone()
    }

    fn pass_boundary_metrics(&self, pass_metrics: MockerMetrics) -> MockerMetrics {
        pass_metrics
    }

    fn live_request_residency(&self, _request_id: Uuid) -> Option<RequestResidency> {
        if self.metrics.running_requests > 0 {
            Some(RequestResidency::Running)
        } else if self.metrics.waiting_requests > 0 {
            Some(RequestResidency::Waiting)
        } else {
            None
        }
    }

    fn execute_live_pass(&mut self, _scheduler_start: &Instant) -> LivePassExecution {
        if let Some(execute_count) = &self.execute_count {
            let count = execute_count.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some((target, cancel_token)) = &self.cancel_after_execute
                && count == *target
            {
                cancel_token.cancel();
            }
        }
        LivePassExecution {
            pass: pass(),
            duration: self.pass_duration,
        }
    }

    fn output_delivery_failed(&mut self, _signals: Vec<OutputSignal>) {
        self.publish_kv(3);
    }
}

fn pass() -> EnginePassResult {
    let request_id = Uuid::from_u128(1);
    EnginePassResult {
        end_ms: 1.0,
        completed_requests: 1,
        output_signals: vec![OutputSignal {
            uuid: request_id,
            token_id: None,
            completed: true,
            rejected: false,
            handoff_delay_ms: None,
        }],
        admissions: vec![AdmissionEvent {
            uuid: request_id,
            reused_input_tokens: 0,
        }],
        lifecycle_events: vec![SchedulerLifecycleEvent::DestinationReserved {
            handoff_id: HandoffId::from(Uuid::from_u128(2)),
            request_id,
            transferable_prompt_tokens: 4,
        }],
        mocker_metrics: MockerMetrics::default(),
        router_event_visibility: RouterEventVisibility::PassEnd,
        kv_events: Vec::new(),
        fpm: Some(ForwardPassSnapshot::default()),
        accept_length_output_tokens: 1,
        accept_length_decode_forwards: 1,
    }
}

fn publisher(
    output_tx: mpsc::UnboundedSender<Vec<OutputSignal>>,
    captured: DeferredKvPublishBuffer,
    log: Arc<Mutex<Vec<PublishedEffect>>>,
) -> LiveEffectsPublisher {
    publisher_with_metrics(output_tx, captured, log, MockerMetrics::default())
}

fn publisher_with_metrics(
    output_tx: mpsc::UnboundedSender<Vec<OutputSignal>>,
    captured: DeferredKvPublishBuffer,
    log: Arc<Mutex<Vec<PublishedEffect>>>,
    metrics: MockerMetrics,
) -> LiveEffectsPublisher {
    let (admission_tx, _admission_rx) = mpsc::unbounded_channel();
    let (lifecycle_tx, _lifecycle_rx) = mpsc::channel(4);
    let (metrics_tx, _metrics_rx) = watch::channel(metrics);
    LiveEffectsPublisher::new(
        Some(output_tx.into()),
        Some(admission_tx),
        lifecycle_tx,
        metrics_tx,
        KvEventPublishers::default(),
        FpmPublisher::default(),
        captured,
    )
    .with_publication_log(log)
}

#[tokio::test]
async fn pass_effects_publish_once_in_boundary_order_and_isolate_midpass_ack() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: false,
        midpass_kv_effects: false,
        metrics: MockerMetrics::default(),
        pass_duration: Duration::from_millis(1),
        execute_count: None,
        cancel_after_execute: None,
    };
    core.publish_kv(1);
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    drop(output_rx);
    let log = Arc::new(Mutex::new(Vec::new()));
    let publisher = publisher(output_tx, captured, log.clone());
    let mut pending = publisher.capture_pass(pass());
    publisher.publish_pass_start(&mut pending);

    let (reply, reply_rx) = tokio::sync::oneshot::channel();
    publisher
        .apply_command(
            &mut core,
            SchedulerCommandEnvelope {
                command: SchedulerCommand::CancelSource {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                },
                reply,
            },
            false,
            1.0,
        )
        .await;
    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[PublishedEffect::Admissions, PublishedEffect::Ack,]
    );

    publisher.publish_pass(&mut core, pending).await;
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Ack,
            PublishedEffect::Kv,
            PublishedEffect::Fpm,
            PublishedEffect::Outputs,
            PublishedEffect::Accounting,
            PublishedEffect::Kv,
            PublishedEffect::Lifecycle,
            PublishedEffect::Metrics,
        ]
    );
}

#[tokio::test]
async fn midpass_cancel_only_publishes_prompt_occupancy_before_pass_end() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: false,
        midpass_kv_effects: true,
        metrics: {
            let mut metrics = MockerMetrics::new(0, 9, 100);
            metrics.running_requests = 1;
            metrics
        },
        pass_duration: Duration::from_millis(1),
        execute_count: None,
        cancel_after_execute: None,
    };
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut published_metrics = MockerMetrics::new(0, 7, 100);
    published_metrics.running_requests = 1;
    published_metrics.waiting_requests = 1;
    let publisher = publisher_with_metrics(output_tx, captured, log.clone(), published_metrics);
    let mut pending = publisher.capture_pass(pass());
    publisher.publish_pass_start(&mut pending);

    let (reply, reply_rx) = tokio::sync::oneshot::channel();
    let request_id = Uuid::from_u128(1);
    // The pending pass moved this request to the core's running queue,
    // while its last published residency is still waiting.
    publisher.set_visible_request_residency(request_id, RequestResidency::Waiting);
    let applied = publisher
        .apply_cancellation(
            &mut core,
            SchedulerCancellationEnvelope {
                request_id,
                discard_pending_output: false,
                reply,
            },
            false,
            1.0,
        )
        .await;
    assert_eq!(applied, Some(SchedulerCommandResult::Applied));
    pending.suppress_request_outputs(request_id);
    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Ack,
            PublishedEffect::Metrics,
        ]
    );
    let metrics = publisher.published_metrics();
    assert_eq!(metrics.active_decode_blocks, 7);
    assert_eq!(metrics.gpu_cache_usage_perc, 0.07);
    assert_eq!(metrics.running_requests, 1);
    assert_eq!(metrics.waiting_requests, 0);

    publisher.publish_pass(&mut core, pending).await;
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Ack,
            PublishedEffect::Metrics,
            PublishedEffect::Kv,
            PublishedEffect::Fpm,
            PublishedEffect::Accounting,
            PublishedEffect::Lifecycle,
            PublishedEffect::Metrics,
        ]
    );
}

#[tokio::test]
async fn noop_midpass_cancel_does_not_publish_pending_completion_metrics() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: false,
        midpass_kv_effects: false,
        // Both requests have completed in the core's captured pass, but
        // that pass has not reached its modeled publication boundary.
        metrics: MockerMetrics::new(0, 0, 100),
        pass_duration: Duration::from_millis(1),
        execute_count: None,
        cancel_after_execute: None,
    };
    let (output_tx, mut output_rx) = mpsc::unbounded_channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut published_metrics = MockerMetrics::new(0, 7, 100);
    published_metrics.running_requests = 2;
    let publisher = publisher_with_metrics(output_tx, captured, log.clone(), published_metrics);
    let mut pending = publisher.capture_pass(pass());
    publisher.publish_pass_start(&mut pending);

    let (reply, reply_rx) = tokio::sync::oneshot::channel();
    let applied = publisher
        .apply_cancellation(
            &mut core,
            SchedulerCancellationEnvelope {
                request_id: Uuid::from_u128(1),
                discard_pending_output: false,
                reply,
            },
            false,
            1.0,
        )
        .await;
    assert_eq!(applied, Some(SchedulerCommandResult::Noop));

    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Noop
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[PublishedEffect::Admissions, PublishedEffect::Ack]
    );
    assert_eq!(publisher.published_metrics().running_requests, 2);
    assert!(output_rx.try_recv().is_err());
}

#[tokio::test]
async fn controlled_pass_start_router_effects_precede_midpass_ack_without_duplicates() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: false,
        midpass_kv_effects: true,
        metrics: MockerMetrics::default(),
        pass_duration: Duration::from_millis(1),
        execute_count: None,
        cancel_after_execute: None,
    };
    core.publish_kv(1);
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let publisher = publisher(output_tx, captured, log.clone());
    let mut pass = pass();
    pass.router_event_visibility = RouterEventVisibility::PassStart;
    let mut pending = publisher.capture_pass(pass);

    publisher.publish_pass_start(&mut pending);
    let (reply, reply_rx) = tokio::sync::oneshot::channel();
    publisher
        .apply_command(
            &mut core,
            SchedulerCommandEnvelope {
                command: SchedulerCommand::CancelSource {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                },
                reply,
            },
            false,
            1.0,
        )
        .await;
    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Kv,
            PublishedEffect::Ack,
        ]
    );

    publisher.publish_pass(&mut core, pending).await;
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Kv,
            PublishedEffect::Ack,
            PublishedEffect::Kv,
            PublishedEffect::Fpm,
            PublishedEffect::Outputs,
            PublishedEffect::Accounting,
            PublishedEffect::Lifecycle,
            PublishedEffect::Metrics,
        ]
    );
}

#[tokio::test]
async fn command_effects_publish_kv_before_ack_then_lifecycle_and_metrics() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: true,
        midpass_kv_effects: false,
        metrics: MockerMetrics::default(),
        pass_duration: Duration::from_millis(1),
        execute_count: None,
        cancel_after_execute: None,
    };
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let publisher = publisher(output_tx, captured, log.clone());
    let (reply, reply_rx) = tokio::sync::oneshot::channel();

    publisher
        .apply_command(
            &mut core,
            SchedulerCommandEnvelope {
                command: SchedulerCommand::ActivateDestination {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                },
                reply,
            },
            true,
            1.0,
        )
        .await;

    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Kv,
            PublishedEffect::Ack,
            PublishedEffect::Lifecycle,
            PublishedEffect::Metrics,
        ]
    );
}

#[tokio::test]
async fn cancellation_stops_a_nonempty_zero_duration_progress_loop() {
    let cancel_token = CancellationToken::new();
    let execute_count = Arc::new(AtomicUsize::new(0));
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(false, false);
    let metrics = MockerMetrics {
        running_requests: 1,
        ..Default::default()
    };
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: false,
        midpass_kv_effects: false,
        metrics,
        pass_duration: Duration::ZERO,
        execute_count: Some(Arc::clone(&execute_count)),
        cancel_after_execute: Some((3, cancel_token.clone())),
    };
    let (request_tx, request_rx) = mpsc::unbounded_channel();
    let (command_tx, command_rx) = mpsc::channel(1);
    let (cancellation_tx, cancellation_rx) = mpsc::channel(1);
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let publisher = publisher(output_tx, captured, Arc::new(Mutex::new(Vec::new())));

    run_live_scheduler(
        &mut core,
        request_rx,
        command_rx,
        cancellation_rx,
        publisher,
        cancel_token,
    )
    .await;

    assert_eq!(execute_count.load(Ordering::Relaxed), 3);
    drop((request_tx, command_tx, cancellation_tx));
}
