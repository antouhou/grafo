use super::TextureUpdatePublisher;
use std::sync::{Arc, Barrier};
use std::thread;

#[test]
fn cloned_publishers_coalesce_changes_for_independent_subscriptions() {
    let publisher = TextureUpdatePublisher::default();
    publisher.publish(99);
    let mut first = publisher.subscribe();
    let mut second = publisher.subscribe();
    let cloned_publisher = publisher.clone();
    publisher.publish(1);
    cloned_publisher.publish(1);
    cloned_publisher.publish(2);

    let mut updates = Vec::new();
    first.drain_into(&mut updates);
    updates.sort_unstable();
    assert_eq!(updates, [1, 2]);
    first.drain_into(&mut updates);
    assert!(updates.is_empty());

    let mut later = publisher.subscribe();
    publisher.publish(3);
    second.drain_into(&mut updates);
    updates.sort_unstable();
    assert_eq!(updates, [1, 2, 3]);
    first.drain_into(&mut updates);
    assert_eq!(updates, [3]);
    later.drain_into(&mut updates);
    assert_eq!(updates, [3]);
}

#[test]
fn concurrent_publication_and_consumption_preserve_every_change() {
    let publisher = TextureUpdatePublisher::default();
    let mut subscription = publisher.subscribe();
    let barrier = Arc::new(Barrier::new(2));
    let mut updates = Vec::new();
    thread::scope(|scope| {
        let producer_barrier = barrier.clone();
        let producer = publisher.clone();
        scope.spawn(move || {
            producer_barrier.wait();
            for texture_id in 0..256 {
                producer.publish(texture_id);
            }
        });
        barrier.wait();
        subscription.drain_into(&mut updates);
    });
    let mut remaining = Vec::new();
    subscription.drain_into(&mut remaining);
    updates.extend(remaining);
    updates.sort_unstable();
    assert_eq!(updates, (0..256).collect::<Vec<_>>());
}

#[test]
fn dropped_subscriptions_release_pending_changes_and_are_removed() {
    let publisher = TextureUpdatePublisher::default();
    let subscription = publisher.subscribe();
    let pending = Arc::downgrade(&subscription.pending);
    publisher.publish(1);
    drop(subscription);
    assert!(pending.upgrade().is_none());
    publisher.publish(2);
    assert!(publisher.subscriptions.lock().unwrap().is_empty());
}
