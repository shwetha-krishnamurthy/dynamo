// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::common::handoff::HandoffId;
use crate::common::protocols::EngineType;

fn test_route(
    client_id: Uuid,
    scheduler_id: Uuid,
    output_tx: mpsc::Sender<OutputSignal>,
) -> Arc<RequestRoute> {
    Arc::new(RequestRoute::new(client_id, scheduler_id, output_tx))
}

fn register_route(routes: &RequestRoutes, route: &Arc<RequestRoute>) {
    assert!(
        routes
            .by_client
            .insert(route.client_id, Arc::clone(route))
            .is_none()
    );
    assert!(
        routes
            .by_scheduler
            .insert(route.scheduler_id, Arc::clone(route))
            .is_none()
    );
}

fn args(engine_type: EngineType) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(engine_type)
        .block_size(4)
        .num_gpu_blocks(128)
        .max_num_seqs(Some(8))
        .max_num_batched_tokens(Some(64))
        .speedup_ratio(1000.0)
        .dp_size(1)
        .build()
        .unwrap()
}

#[tokio::test]
async fn streams_planned_tokens_to_the_owning_request() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start(args(engine_type), 0).unwrap();
        let uuid = Uuid::from_u128(1);
        let mut request = engine
            .submit(DirectRequest {
                tokens: vec![1, 2, 3],
                max_output_tokens: 3,
                output_token_ids: Some(vec![41, 42, 43]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();

        let mut outputs = Vec::new();
        while let Some(signal) = request.recv().await {
            outputs.push((signal.uuid, signal.token_id, signal.completed));
            if signal.completed {
                break;
            }
        }
        assert_eq!(
            outputs,
            vec![
                (uuid, Some(41), false),
                (uuid, Some(42), false),
                (uuid, Some(43), true),
            ]
        );
        assert!(request.recv().await.is_none());
        assert_eq!(engine.active_request_count(), 0);
    }
}

#[tokio::test]
async fn dropping_engine_closes_outstanding_request_streams() {
    let engine = LiveEngine::start(args(EngineType::Vllm), 0).unwrap();
    let mut request = engine
        .submit(DirectRequest {
            tokens: vec![1; 256],
            max_output_tokens: 10_000,
            uuid: Some(Uuid::from_u128(6)),
            ..Default::default()
        })
        .await
        .unwrap();

    drop(engine);
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while request.recv().await.is_some() {}
    })
    .await
    .expect("engine shutdown should close every outstanding output route");
}

#[tokio::test]
async fn cancellation_does_not_commit_a_cobatched_pass_early() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let mut slow_args = args(engine_type);
        slow_args.speedup_ratio = 0.001;
        let engine = LiveEngine::start(slow_args, 0).unwrap();
        let (first, second) = tokio::join!(
            engine.submit(DirectRequest {
                tokens: vec![1; 256],
                max_output_tokens: 10_000,
                uuid: Some(Uuid::from_u128(2)),
                ..Default::default()
            }),
            engine.submit(DirectRequest {
                tokens: vec![1; 256],
                max_output_tokens: 10_000,
                uuid: Some(Uuid::from_u128(3)),
                ..Default::default()
            })
        );
        let first = first.unwrap();
        let mut second = second.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Boundary-only work remains on the ordinary bounded lane while
        // cancellation is serviced independently during the long pass.
        let (deferred_reply, _deferred_ack) = oneshot::channel();
        engine
            .inner
            .command_tx
            .send(SchedulerCommandEnvelope {
                command: SchedulerCommand::CancelSource {
                    handoff_id: HandoffId::from(Uuid::from_u128(20)),
                },
                reply: deferred_reply,
            })
            .await
            .unwrap();
        let first_id = first.id();
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), engine.cancel(first_id))
                .await
                .expect("cancellation should not wait for the modeled pass")
                .unwrap()
        );
        drop(first);

        let mut metrics = engine.metrics_receiver();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let snapshot = metrics.borrow_and_update().clone();
                if snapshot.running_requests + snapshot.waiting_requests == 1 {
                    break;
                }
                metrics.changed().await.unwrap();
            }
        })
        .await
        .expect("cancellation should publish prompt occupancy promptly");
        assert_eq!(engine.active_request_count(), 1);

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), second.recv())
                .await
                .is_err(),
            "cancelling one request must not publish a co-batched output early"
        );
        drop(second);
        drop(engine);
    }
}

#[tokio::test]
async fn duplicate_request_id_does_not_replace_the_original_stream() {
    let engine = LiveEngine::start(args(EngineType::Vllm), 0).unwrap();
    let uuid = Uuid::from_u128(3);
    let original = engine
        .submit(DirectRequest {
            tokens: vec![1, 2, 3],
            max_output_tokens: 1_000,
            uuid: Some(uuid),
            ..Default::default()
        })
        .await
        .unwrap();
    let duplicate = engine
        .submit(DirectRequest {
            tokens: vec![4, 5, 6],
            max_output_tokens: 1,
            uuid: Some(uuid),
            ..Default::default()
        })
        .await;
    let error = match duplicate {
        Ok(_) => panic!("duplicate request ID must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("already active"));
    assert_eq!(engine.active_request_count(), 1);
    original.cancel().await.unwrap();
    assert_eq!(engine.active_request_count(), 0);
}

#[tokio::test]
async fn cancelled_pass_output_does_not_reach_reused_request_id() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let mut timed_args = args(engine_type);
        timed_args.speedup_ratio = 0.1;
        let engine = LiveEngine::start(timed_args, 0).unwrap();
        let uuid = Uuid::from_u128(8);
        let old = engine
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 100,
                output_token_ids: Some(vec![11; 100]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), engine.cancel(uuid))
                .await
                .expect("old request cancellation should be observed during the pass")
                .unwrap()
        );
        drop(old);

        let mut replacement = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            engine.submit(DirectRequest {
                tokens: vec![2],
                max_output_tokens: 1,
                output_token_ids: Some(vec![22]),
                uuid: Some(uuid),
                ..Default::default()
            }),
        )
        .await
        .expect("replacement should be admitted after the pending pass boundary")
        .unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(3), replacement.recv())
            .await
            .expect("replacement should produce its planned token")
            .unwrap();
        assert_eq!(output.token_id, Some(22));
        assert!(output.completed);
        assert!(replacement.recv().await.is_none());
        assert_eq!(engine.active_request_count(), 0);
    }
}

#[tokio::test]
async fn dropped_terminal_pass_output_does_not_reach_reused_request_id() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let mut timed_args = args(engine_type);
        timed_args.speedup_ratio = 0.1;
        let engine = LiveEngine::start(timed_args, 0).unwrap();
        let uuid = Uuid::from_u128(9);
        let old = engine
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 1,
                output_token_ids: Some(vec![11]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        drop(old);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while engine.active_request_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped terminal stream should retire after cancellation acknowledgement");

        let mut replacement = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            engine.submit(DirectRequest {
                tokens: vec![2],
                max_output_tokens: 1,
                output_token_ids: Some(vec![22]),
                uuid: Some(uuid),
                ..Default::default()
            }),
        )
        .await
        .expect("replacement should be admitted after the pending pass boundary")
        .unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(3), replacement.recv())
            .await
            .expect("replacement should produce its planned token")
            .unwrap();
        assert_eq!(output.token_id, Some(22));
        assert!(output.completed);
        assert!(replacement.recv().await.is_none());
        assert_eq!(engine.active_request_count(), 0);
    }
}

#[test]
fn start_without_a_tokio_runtime_returns_an_error() {
    let result = LiveEngine::start(args(EngineType::Vllm), 0);
    let error = match result {
        Ok(_) => panic!("LiveEngine unexpectedly started without a Tokio runtime"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("active Tokio runtime"));
}

#[tokio::test]
async fn internal_scheduler_ids_isolate_queued_output_from_reused_client_ids() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let mut engine_args = args(engine_type);
        engine_args.speedup_ratio = 0.01;
        let (gate_tx, gate_rx) = watch::channel(false);
        let engine = LiveEngine::start_with_output_gate(
            engine_args,
            0,
            Some(gate_rx),
            MAX_BUFFERED_OUTPUT_SIGNALS,
        )
        .unwrap();
        let uuid = Uuid::from_u128(12);
        let old = engine
            .submit(DirectRequest {
                tokens: vec![1; 64],
                max_output_tokens: 1,
                output_token_ids: Some(vec![11]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();

        let mut metrics = engine.metrics_receiver();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                let current = metrics.borrow_and_update().clone();
                if current.running_requests == 0 && current.waiting_requests == 0 {
                    break;
                }
                metrics.changed().await.unwrap();
            }
        })
        .await
        .expect("old terminal output should be queued at the dispatcher boundary");

        drop(old);
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while engine.active_request_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped request should retire without waiting for queued output");

        let mut replacement = engine
            .submit(DirectRequest {
                tokens: vec![2],
                max_output_tokens: 1,
                output_token_ids: Some(vec![22]),
                uuid: Some(uuid),
                ..Default::default()
            })
            .await
            .unwrap();
        gate_tx.send_replace(true);
        let output = tokio::time::timeout(std::time::Duration::from_secs(3), replacement.recv())
            .await
            .expect("replacement output should be dispatched")
            .unwrap();
        assert_eq!(output.token_id, Some(22));
        assert_eq!(output.uuid, uuid);
        assert!(output.completed);
    }
}

#[tokio::test]
async fn slow_reader_does_not_stall_an_unrelated_request() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start_with_output_gate(args(engine_type), 0, None, 4).unwrap();
        let mut slow = engine
            .submit(DirectRequest {
                tokens: vec![1],
                // Explicit plans are authoritative in both scheduler cores;
                // the live adapter must reserve their effective length.
                max_output_tokens: 1,
                output_token_ids: Some(vec![7; 3]),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            })
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let mut fast = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            engine.submit(DirectRequest {
                tokens: vec![2],
                max_output_tokens: 1,
                output_token_ids: Some(vec![22]),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            }),
        )
        .await
        .expect("an unrelated request should use the remaining output budget")
        .unwrap();
        let fast_output = tokio::time::timeout(std::time::Duration::from_secs(1), fast.recv())
            .await
            .expect("unrelated request should not wait for the slow reader")
            .unwrap();
        assert_eq!(fast_output.token_id, Some(22));
        assert!(fast_output.completed);

        let received = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            let mut received = 0;
            while let Some(signal) = slow.recv().await {
                received += 1;
                if signal.completed {
                    break;
                }
            }
            received
        })
        .await
        .expect("slow reader should resume and receive the full response");
        assert_eq!(received, 3);
        assert!(slow.recv().await.is_none());
        assert_eq!(engine.active_request_count(), 0);

        let metrics = engine.metrics_receiver().borrow().clone();
        assert_eq!(metrics.running_requests, 0);
        assert_eq!(metrics.waiting_requests, 0);
    }
}

#[tokio::test]
async fn empty_effective_output_is_rejected_before_route_registration() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start(args(engine_type), 0).unwrap();
        let error = engine
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 4,
                output_token_ids: Some(Vec::new()),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            })
            .await
            .err()
            .expect("empty explicit output plan should be rejected");
        assert!(error.to_string().contains("at least one output token"));
        assert_eq!(engine.active_request_count(), 0);
    }
}

#[tokio::test]
async fn output_budget_bounds_slow_reader_memory_until_its_stream_is_released() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let engine = LiveEngine::start_with_output_gate(args(engine_type), 0, None, 4).unwrap();
        let slow = engine
            .submit(DirectRequest {
                tokens: vec![1],
                max_output_tokens: 4,
                output_token_ids: Some(vec![7; 4]),
                uuid: Some(Uuid::new_v4()),
                ..Default::default()
            })
            .await
            .unwrap();

        let pending = engine.submit(DirectRequest {
            tokens: vec![2],
            max_output_tokens: 1,
            output_token_ids: Some(vec![22]),
            uuid: Some(Uuid::new_v4()),
            ..Default::default()
        });
        tokio::pin!(pending);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut pending)
                .await
                .is_err(),
            "submission should wait rather than exceed the global output budget"
        );

        drop(slow);
        let mut admitted = tokio::time::timeout(std::time::Duration::from_secs(1), pending)
            .await
            .expect("dropping the buffered stream should release its output budget")
            .unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(1), admitted.recv())
            .await
            .expect("request admitted after budget release should make progress")
            .unwrap();
        assert_eq!(output.token_id, Some(22));
        assert!(output.completed);
    }
}

#[tokio::test]
async fn cancellation_cleanup_survives_a_cancelled_caller() {
    let client_id = Uuid::from_u128(4);
    let scheduler_id = Uuid::from_u128(104);
    let routes = Routes::default();
    let (output_tx, mut output_rx) = mpsc::channel(4);
    let route = test_route(client_id, scheduler_id, output_tx);
    route.activate();
    register_route(&routes, &route);
    let (command_tx, mut command_rx) = mpsc::channel(1);

    let cancel = tokio::spawn(await_cancellation(spawn_cancellation(
        &Handle::current(),
        command_tx,
        Arc::clone(&routes),
        Arc::clone(&route),
        false,
    )));
    let command = command_rx.recv().await.unwrap();
    assert_eq!(command.request_id, scheduler_id);

    cancel.abort();
    assert!(cancel.await.unwrap_err().is_cancelled());
    command
        .reply
        .send(Ok(crate::scheduler::SchedulerCommandEffects::new(
            SchedulerCommandResult::Applied,
        )))
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !routes.by_client.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("owned cancellation task should finish route cleanup");
    assert!(output_rx.recv().await.is_none());
}

#[tokio::test]
async fn dropped_request_holds_its_client_id_until_cancel_ack() {
    let client_id = Uuid::from_u128(7);
    let scheduler_id = Uuid::from_u128(107);
    let routes = Routes::default();
    let (output_tx, output_rx) = mpsc::channel(4);
    let route = test_route(client_id, scheduler_id, output_tx);
    route.activate();
    register_route(&routes, &route);
    let (command_tx, mut command_rx) = mpsc::channel(1);
    let output_budget = Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap();
    let request = LiveRequest {
        client_id,
        rx: output_rx,
        route: Arc::downgrade(&route),
        routes: Arc::clone(&routes),
        cancellation_tx: command_tx,
        runtime: Handle::current(),
        _output_budget: output_budget,
    };

    drop(request);
    let command = command_rx.recv().await.unwrap();
    assert_eq!(command.request_id, scheduler_id);
    assert!(matches!(
        routes.by_client.entry(client_id),
        Entry::Occupied(_)
    ));

    command
        .reply
        .send(Ok(crate::scheduler::SchedulerCommandEffects::new(
            SchedulerCommandResult::Noop,
        )))
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !routes.by_client.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cancel acknowledgement should retire the tombstone");

    let (replacement_tx, _replacement_rx) = mpsc::channel(4);
    let replacement = test_route(client_id, Uuid::from_u128(207), replacement_tx);
    register_route(&routes, &replacement);
    let next_command =
        tokio::time::timeout(std::time::Duration::from_millis(20), command_rx.recv()).await;
    assert!(
        !matches!(next_command, Ok(Some(_))),
        "retired cleanup must not enqueue a cancellation for the replacement"
    );
    replacement.shutdown();
    assert!(remove_route(&routes, &replacement));
}

#[tokio::test]
async fn cancel_waits_for_admission_and_preserves_a_noop_terminal_route() {
    let client_id = Uuid::from_u128(5);
    let scheduler_id = Uuid::from_u128(105);
    let routes = Routes::default();
    let (output_tx, mut output_rx) = mpsc::channel(4);
    let route = test_route(client_id, scheduler_id, output_tx);
    register_route(&routes, &route);
    let (command_tx, mut command_rx) = mpsc::channel(1);

    let cancel = spawn_cancellation(
        &Handle::current(),
        command_tx,
        Arc::clone(&routes),
        Arc::clone(&route),
        false,
    );

    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), command_rx.recv())
            .await
            .is_err(),
        "cancellation must not race ahead of scheduler admission"
    );
    route.activate();

    let command = command_rx.recv().await.unwrap();
    assert_eq!(command.request_id, scheduler_id);
    command
        .reply
        .send(Ok(crate::scheduler::SchedulerCommandEffects::new(
            SchedulerCommandResult::Noop,
        )))
        .unwrap();
    assert!(!await_cancellation(cancel).await.unwrap());
    assert_eq!(routes.by_client.len(), 1);

    route
        .send_output(OutputSignal {
            uuid: client_id,
            token_id: None,
            completed: true,
            rejected: false,
            handoff_delay_ms: None,
        })
        .await;
    assert!(route.observe_terminal());
    assert!(remove_route(&routes, &route));
    assert!(output_rx.recv().await.unwrap().completed);
    assert!(output_rx.recv().await.is_none());
}
